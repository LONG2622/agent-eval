"""Tests for the FastAPI server (agent_eval.server).

The server's ``state`` module creates a real ``JSONLStorage`` pointing at
``./outputs`` at import time. To avoid polluting real outputs, the ``server``
fixture below redirects the storage directories to a temporary location via
env vars (``OUTPUT_DIR``/``TRACE_DIR``/``RUN_DIR``/``EVAL_DIR`` are consumed
by ``configs/default.yaml``) *before* importing the app, and additionally
rebinds the module-level storage singletons to temp-based instances.

Endpoints that would call the LLM (POST /api/runs happy path, judge on a real
run) are only exercised through validation/error paths that never reach the
LLM.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent_eval import __version__

# Unique ids that must NOT exist in any storage
UNKNOWN_RUN_ID = "nonexistent_run_xyz_42"
UNKNOWN_ANNOTATION_ID = "nonexistent_annotation_xyz_42"

# Module-level memo so the seeded data is written only once per test module
_SEEDED: dict = {}


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """TestClient with all server storage redirected to a temp directory.

    Env vars are set before ``agent_eval.server.app`` is imported so that
    ``server.state`` builds its JSONLStorage inside the temp dir. The module
    level singletons are also explicitly rebound for robustness.
    """
    mp = pytest.MonkeyPatch()
    tmp_root = tmp_path_factory.mktemp("server_outputs")
    mp.setenv("OUTPUT_DIR", str(tmp_root / "outputs"))
    mp.setenv("TRACE_DIR", str(tmp_root / "outputs" / "traces"))
    mp.setenv("RUN_DIR", str(tmp_root / "outputs" / "runs"))
    mp.setenv("EVAL_DIR", str(tmp_root / "outputs" / "evaluations"))

    from agent_eval import config as config_module
    from agent_eval.trace.storage import JSONLStorage

    config_module.reset_config()
    config_module.load_config(force_reload=True)

    from agent_eval.server import state as server_state
    from agent_eval.server.routes import api as server_api

    redirected = JSONLStorage()
    mp.setattr(server_state, "_storage", redirected, raising=True)
    mp.setattr(server_api, "_storage", redirected, raising=True)
    # Force get_engine() to rebuild against the redirected storage
    mp.setattr(server_state, "_engine", None, raising=True)

    from agent_eval.server.app import app

    # raise_server_exceptions=False so that internal errors (e.g. pydantic
    # validation raised inside a handler) surface as HTTP 500 responses
    # instead of blowing up the test client.
    client = TestClient(app, raise_server_exceptions=False)
    yield client

    mp.undo()
    config_module.reset_config()


@pytest.fixture
def seeded(server, sample_run, sample_spans):
    """Seed the server's (temp) storage with one success run + spans and one
    failed run. Executed only once per module thanks to the memo dict."""
    from agent_eval.server.routes import api as server_api
    from agent_eval.trace.models import AnnotationRecord, RunRecord, RunStatus

    if not _SEEDED:
        storage = server_api._storage
        storage.save_run(sample_run)
        storage.append_spans(sample_spans)

        failed_run = RunRecord(
            run_id="test_run_failed_001",
            task_id="task_002",
            agent_name="react_agent",
            input_text="This task is expected to fail",
            status=RunStatus.FAILED,
            error_message="API request timed out after retries",
            total_steps=2,
            total_latency_ms=3000,
        )
        storage.save_run(failed_run)

        # Baseline annotation so annotation-dependent endpoints (comparison
        # report) work even when TestAnnotationAPI did not run first.
        baseline_ann = AnnotationRecord(
            run_id=sample_run.run_id,
            annotator="baseline",
            score=4,
            labels=["correct"],
            comment="Good answer",
        )
        storage.save_annotation(baseline_ann)

        _SEEDED.update(
            success_run_id=sample_run.run_id,
            failed_run_id=failed_run.run_id,
        )
    return dict(_SEEDED)


# ============================================================
# Health / Config / Models
# ============================================================


class TestHealthAPI:
    def test_health_returns_ok(self, server, seeded):
        resp = server.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["storage_backend"] == "jsonl"
        assert isinstance(data["total_runs"], int)
        assert data["total_runs"] >= 2
        assert isinstance(data["default_model"], str) and data["default_model"]

    def test_health_version_matches_package(self, server, seeded):
        resp = server.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["version"] == __version__

    def test_get_config(self, server):
        resp = server.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        for key in ("agent", "llm", "pricing", "storage", "evaluation"):
            assert key in data
        assert isinstance(data["storage"]["output_dir"], str)
        assert data["llm"]["default_model"]

    def test_get_config_reports_temp_storage(self, server):
        # The server fixture redirected storage; config must reflect that.
        resp = server.get("/api/config")
        assert resp.status_code == 200
        out_dir = resp.json()["storage"]["output_dir"]
        assert "server_outputs" in out_dir

    def test_list_models(self, server):
        resp = server.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == len(data["models"])
        assert isinstance(data["default_model"], str)
        for m in data["models"]:
            for field in (
                "id",
                "display_name",
                "model",
                "provider",
                "description",
                "supports_function_calling",
                "supports_chinese",
                "is_default",
            ):
                assert field in m

    def test_list_models_default_flag_consistent(self, server):
        resp = server.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        defaults = [m for m in data["models"] if m["is_default"]]
        if defaults:
            assert all(m["model"] == data["default_model"] or m["id"] == data["default_model"] for m in defaults)


# ============================================================
# Runs API
# ============================================================


class TestRunsAPI:
    def test_list_runs_ok(self, server, seeded):
        resp = server.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data and "runs" in data
        assert data["total"] == len(data["runs"])
        assert data["total"] >= 2

    def test_list_runs_entry_shape(self, server, seeded):
        resp = server.get("/api/runs")
        assert resp.status_code == 200
        for entry in resp.json()["runs"]:
            for key in (
                "run_id",
                "task_id",
                "agent_name",
                "status",
                "input_text",
                "final_output",
                "total_latency_ms",
                "total_steps",
                "tokens",
                "total_cost",
                "started_at",
            ):
                assert key in entry
            assert set(entry["tokens"].keys()) == {"prompt_tokens", "completion_tokens", "total_tokens"}

    def test_list_runs_limit(self, server, seeded):
        resp = server.get("/api/runs", params={"limit": 1})
        assert resp.status_code == 200
        assert len(resp.json()["runs"]) <= 1

    def test_list_runs_limit_validation(self, server):
        resp = server.get("/api/runs", params={"limit": 0})
        assert resp.status_code == 422

    def test_list_runs_task_id_filter(self, server, seeded):
        resp = server.get("/api/runs", params={"task_id": "task_001"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert all(r["task_id"] == "task_001" for r in data["runs"])

    def test_list_runs_status_filter(self, server, seeded):
        resp = server.get("/api/runs", params={"status": "success"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert all(r["status"] == "success" for r in data["runs"])

    def test_get_run_not_found(self, server):
        resp = server.get(f"/api/runs/{UNKNOWN_RUN_ID}")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_get_run_seeded(self, server, seeded):
        run_id = seeded["success_run_id"]
        resp = server.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {"run", "spans", "evaluation"}
        assert data["run"]["run_id"] == run_id
        assert data["run"]["status"] == "success"
        assert len(data["spans"]) == 4
        assert isinstance(data["evaluation"], list)

    def test_get_run_trace_seeded(self, server, seeded):
        run_id = seeded["success_run_id"]
        resp = server.get(f"/api/runs/{run_id}/trace")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run"]["run_id"] == run_id
        assert data["total_steps"] == 4
        assert len(data["steps"]) == 4
        first = data["steps"][0]
        assert first["step_index"] == 0
        assert first["total_spans"] >= 1
        assert first["has_error"] is False
        assert "latency_ms" in first

    def test_get_run_trace_not_found(self, server):
        resp = server.get(f"/api/runs/{UNKNOWN_RUN_ID}/trace")
        assert resp.status_code == 404

    def test_evaluate_unknown_run_not_found(self, server):
        # engine.evaluate_run raises KeyError -> endpoint maps to 404.
        # No LLM is called because the run does not exist.
        resp = server.post(f"/api/runs/{UNKNOWN_RUN_ID}/evaluate")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_judge_unknown_run_not_found(self, server):
        # Judge endpoint checks run existence BEFORE creating LLMJudgeEvaluator,
        # so no LLM call happens for a fake run id.
        resp = server.post(f"/api/judge/{UNKNOWN_RUN_ID}")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_eval_batch_validation_error(self, server):
        # Missing required dataset_path -> request validation error (422).
        resp = server.post("/api/eval", json={})
        assert resp.status_code == 422

    def test_eval_batch_dataset_not_found(self, server):
        # Dataset existence is checked before any runner/LLM work.
        resp = server.post("/api/eval", json={"dataset_path": "no_such_dir/does_not_exist.jsonl"})
        assert resp.status_code == 404
        assert resp.json()["detail"].startswith("Dataset not found")

    def test_eval_get_method_not_allowed(self, server):
        resp = server.get("/api/eval")
        assert resp.status_code == 405

    def test_create_run_validation_error(self, server):
        # Missing required `task` field -> 422 raised by request validation,
        # so the handler (and the LLM) is never reached.
        resp = server.post("/api/runs", json={})
        assert resp.status_code == 422

    def test_compare_dataset_not_found(self, server):
        # A/B compare checks dataset existence before any LLM call.
        resp = server.post("/api/compare", json={"dataset_path": "no_such_dir/does_not_exist.jsonl"})
        assert resp.status_code == 404
        assert resp.json()["detail"].startswith("Dataset not found")

    def test_unknown_api_route_404(self, server):
        resp = server.get("/api/definitely_not_a_route")
        assert resp.status_code == 404


# ============================================================
# Annotations API
# ============================================================


class TestAnnotationAPI:
    def test_post_annotation_success(self, server, seeded):
        run_id = seeded["success_run_id"]
        resp = server.post(
            f"/api/runs/{run_id}/annotate",
            json={
                "score": 4,
                "labels": ["correct", "complete"],
                "comment": "Good answer",
                "annotator": "api_tester",
            },
        )
        assert resp.status_code == 200
        ann = resp.json()["annotation"]
        assert ann["run_id"] == run_id
        assert ann["score"] == 4
        assert ann["annotator"] == "api_tester"
        assert ann["comment"] == "Good answer"
        assert ann["labels"] == ["correct", "complete"]
        assert ann["annotation_id"]

    def test_post_annotation_unknown_run_404(self, server):
        resp = server.post(
            f"/api/runs/{UNKNOWN_RUN_ID}/annotate",
            json={"score": 3, "annotator": "tester"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_post_annotation_invalid_score_rejected(self, server, seeded):
        # AnnotationRequest accepts any int, but AnnotationRecord enforces
        # ge=1/le=5 -> the handler raises -> 500 (unhandled pydantic error).
        run_id = seeded["success_run_id"]
        resp = server.post(f"/api/runs/{run_id}/annotate", json={"score": 99})
        assert resp.status_code in (422, 500)

    def test_get_run_annotations(self, server, seeded):
        run_id = seeded["success_run_id"]
        resp = server.get(f"/api/runs/{run_id}/annotations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert isinstance(data["annotations"], list)
        assert any(a["annotator"] == "api_tester" and a["score"] == 4 for a in data["annotations"])
        assert any(a["annotator"] == "baseline" for a in data["annotations"])
        for a in data["annotations"]:
            for key in ("annotation_id", "run_id", "annotator", "score", "labels", "comment", "created_at"):
                assert key in a

    def test_get_annotations_list_all(self, server, seeded):
        resp = server.get("/api/annotations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == len(data["annotations"])
        assert data["total"] >= 1
        assert all(a["annotation_id"] for a in data["annotations"])

    def test_get_annotations_limit(self, server, seeded):
        resp = server.get("/api/annotations", params={"limit": 1})
        assert resp.status_code == 200
        assert len(resp.json()["annotations"]) <= 1

    def test_delete_annotation_success(self, server, seeded):
        from agent_eval.server.routes import api as server_api
        from agent_eval.trace.models import AnnotationRecord

        ann = AnnotationRecord(run_id=seeded["success_run_id"], annotator="doomed", score=2)
        server_api._storage.save_annotation(ann)

        resp = server.delete(f"/api/annotations/{ann.annotation_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["annotation_id"] == ann.annotation_id

    def test_delete_annotation_not_found(self, server):
        resp = server.delete(f"/api/annotations/{UNKNOWN_ANNOTATION_ID}")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]


# ============================================================
# Error Classification API
# ============================================================


class TestErrorsAPI:
    def test_error_summary_ok(self, server, seeded):
        resp = server.get("/api/errors/summary")
        assert resp.status_code == 200
        data = resp.json()
        for key in ("total_runs", "total_failed", "total_success", "failure_rate", "by_category", "recent_errors"):
            assert key in data
        assert data["total_runs"] >= 2
        assert data["total_failed"] >= 1
        assert data["total_success"] >= 1
        assert 0.0 <= data["failure_rate"] <= 1.0

    def test_error_summary_counts_consistent(self, server, seeded):
        resp = server.get("/api/errors/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_runs"] == data["total_failed"] + data["total_success"]

    def test_error_summary_classifies_timeout(self, server, seeded):
        resp = server.get("/api/errors/summary")
        assert resp.status_code == 200
        data = resp.json()
        # by_category is serialized as a list of category dicts
        assert isinstance(data["by_category"], list)
        by_code = {c["code"]: c for c in data["by_category"]}
        assert "llm_timeout" in by_code
        assert by_code["llm_timeout"]["count"] >= 1
        assert by_code["llm_timeout"]["label"] == "LLM Timeout"
        timeout_errors = [e for e in data["recent_errors"] if e["run_id"] == seeded["failed_run_id"]]
        assert len(timeout_errors) == 1
        assert timeout_errors[0]["category_code"] == "llm_timeout"

    def test_error_summary_recent_error_shape(self, server, seeded):
        resp = server.get("/api/errors/summary")
        assert resp.status_code == 200
        for e in resp.json()["recent_errors"]:
            for key in ("run_id", "task", "agent_name", "category_code", "category_label", "error_message", "latency_ms", "steps"):
                assert key in e


# ============================================================
# Comparison Report API
# ============================================================


class TestComparisonAPI:
    def test_comparison_report_ok(self, server, seeded):
        resp = server.get("/api/comparison/report")
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data and "items" in data
        assert isinstance(data["items"], list)
        assert isinstance(data["summary"], dict)

    def test_comparison_report_summary_shape(self, server, seeded):
        resp = server.get("/api/comparison/report")
        assert resp.status_code == 200
        summary = resp.json()["summary"]
        for key in ("total_runs_with_annotations", "total_runs_with_auto_eval", "total_runs_both"):
            assert key in summary
        # One annotated run (test_run_001) + auto-eval is triggered on the fly
        # for successful runs by run_comparison() -> both counts are 1.
        assert summary["total_runs_with_annotations"] == 1
        assert summary["total_runs_with_auto_eval"] == 1
        assert summary["total_runs_both"] == 1
        # Human score 4 normalized to (4-1)/4 = 0.75
        assert summary["human_mean"] == 0.75
        # Pearson correlation needs >= 2 paired samples
        assert summary["correlation"] is None
        assert summary["mae"] is not None

    def test_comparison_report_items_per_run(self, server, seeded):
        resp = server.get("/api/comparison/report")
        assert resp.status_code == 200
        items = resp.json()["items"]
        # One item per seeded run (items are built for every run)
        assert len(items) == 2
        by_run = {i["run_id"]: i for i in items}

        success_item = by_run[seeded["success_run_id"]]
        assert success_item["status"] == "success"
        assert success_item["human_score_raw"] == 4
        assert success_item["human_score"] == 0.75
        # Annotator is the latest annotation ("baseline" seeded or "api_tester"
        # created by TestAnnotationAPI earlier in file order)
        assert success_item["annotator"] in {"baseline", "api_tester"}
        assert success_item["auto_overall_score"] is not None
        assert success_item["discrepancy"] is not None

        failed_item = by_run[seeded["failed_run_id"]]
        assert failed_item["status"] == "failed"
        assert failed_item["human_score"] is None
        assert failed_item["auto_overall_score"] is None
        assert failed_item["discrepancy"] is None


# ============================================================
# HTML Pages
# ============================================================


class TestPages:
    @pytest.mark.parametrize(
        "path",
        ["/", "/dashboard", "/chat", "/errors", "/compare"],
    )
    def test_page_returns_html(self, server, path):
        resp = server.get(path)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert resp.text.strip() != ""

    def test_trace_page_renders_for_unknown_run(self, server):
        # pages.py renders the template regardless of run existence;
        # the frontend fetches data via /api/runs/{run_id}.
        resp = server.get(f"/trace/{UNKNOWN_RUN_ID}")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")

    def test_annotate_page_renders_for_unknown_run(self, server):
        resp = server.get(f"/annotate/{UNKNOWN_RUN_ID}")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")

    def test_unknown_page_404(self, server):
        resp = server.get("/no/such/page")
        assert resp.status_code == 404


# ============================================================
# WebSocket
# ============================================================


class TestWebSocket:
    def test_ws_echo_ack(self, server):
        with server.websocket_connect("/ws") as ws:
            ws.send_text("ping")
            data = ws.receive_json()
        assert data == {"type": "ack", "data": "ping"}

    def test_ws_echo_multiple_messages_in_order(self, server):
        with server.websocket_connect("/ws") as ws:
            ws.send_text("hello")
            first = ws.receive_json()
            ws.send_text("world")
            second = ws.receive_json()
        assert first == {"type": "ack", "data": "hello"}
        assert second == {"type": "ack", "data": "world"}
