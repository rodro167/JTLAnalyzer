"""Tests for the REST API layer.

Uses FastAPI's synchronous ``TestClient``. No LLM provider is mocked anywhere in
this module: the API path calls the specialist agents directly and never invokes
one, so any provider interaction here would signal a regression in that design.
"""

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from jtl_analyzer.api import dependencies
from jtl_analyzer.api.main import app
from jtl_analyzer.config import ApiConfig

FIXTURES = Path(__file__).parent / "fixtures"

# Bytes that are neither valid CSV-with-JTL-columns nor valid XML.
GARBAGE_BYTES = b"\x00\x01\x02 this is not a jtl file at all \xff\xfe\n" * 20


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Return a TestClient that surfaces 500 responses instead of re-raising.

    A handler registered for bare ``Exception`` runs on Starlette's
    ``ServerErrorMiddleware``, which re-raises after building the response.
    ``raise_server_exceptions=False`` is therefore required to observe the 500
    body rather than have the exception propagate into the test.
    """
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _upload(name: str = "multi_feature.jtl") -> dict[str, tuple[str, bytes, str]]:
    """Build a ``files=`` mapping for a fixture in ``tests/fixtures``."""
    return {"file": (name, (FIXTURES / name).read_bytes(), "text/csv")}


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"]

    def test_health_needs_no_llm_credentials(self, client: TestClient) -> None:
        # Guards the load_api_config()/load_config() split: if the API ever
        # started reading provider credentials, importing the app or serving a
        # request would fail in an environment without an API key.
        assert client.get("/health").status_code == 200


class TestAnalyzeEndpoint:
    def test_analyze_returns_full_result(self, client: TestClient) -> None:
        response = client.post("/analyze", files=_upload())
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"stats", "errors", "anomalies", "trends"}

    def test_analyze_reports_are_self_contained(self, client: TestClient) -> None:
        body = client.post("/analyze", files=_upload()).json()
        for key in ("stats", "errors", "anomalies", "trends"):
            assert "dataset_metadata" in body[key], f"{key} lost its metadata"

    def test_analyze_serializes_timestamps_as_iso_8601(self, client: TestClient) -> None:
        metadata = client.post("/analyze", files=_upload()).json()["stats"]["dataset_metadata"]
        assert metadata["start_time"].startswith("2023-11-14T")
        assert "T" in metadata["end_time"]

    def test_analyze_omits_raw_sample_rows(self, client: TestClient) -> None:
        # Only aggregates cross the wire; the DataFrame must never be serialized.
        body = client.post("/analyze", files=_upload()).json()
        assert "data" not in body["stats"]
        assert "dataframe" not in body["stats"]

    def test_analyze_reports_client_filename_not_server_temp_path(
        self, client: TestClient
    ) -> None:
        # The upload is spooled to a temp file for the loader, but that path is
        # meaningless to the client and discloses the server's temp layout.
        body = client.post("/analyze", files=_upload()).json()
        for key in ("stats", "errors", "anomalies", "trends"):
            file_path = body[key]["dataset_metadata"]["file_path"]
            assert file_path == "multi_feature.jtl"
            assert "jtl_upload_" not in file_path

    def test_analyze_with_warmup(self, client: TestClient) -> None:
        without = client.post("/analyze", files=_upload()).json()
        with_warmup = client.post(
            "/analyze", files=_upload(), data={"warmup_seconds": "1"}
        ).json()

        assert with_warmup["stats"]["total_count"] < without["stats"]["total_count"]

        metadata = with_warmup["stats"]["dataset_metadata"]
        assert metadata["warmup_seconds"] == 1.0
        # original_row_count records the pre-filter total, so the drop is visible
        # from the response alone.
        assert metadata["original_row_count"] == without["stats"]["total_count"]

    def test_analyze_accepts_xml_format(self, client: TestClient) -> None:
        # Format is sniffed from the file's first bytes, not the filename.
        response = client.post("/analyze", files=_upload("small_clean_xml.jtl"))
        assert response.status_code == 200
        assert response.json()["stats"]["total_count"] > 0

    def test_analyze_invalid_file_returns_400(self, client: TestClient) -> None:
        response = client.post("/analyze", files={"file": ("junk.jtl", GARBAGE_BYTES)})
        assert response.status_code == 400
        body = response.json()
        assert body["error"] == "invalid_jtl"
        assert body["message"]

    def test_analyze_missing_file_returns_422(self, client: TestClient) -> None:
        assert client.post("/analyze").status_code == 422

    def test_analyze_warmup_excluding_all_samples_returns_400(
        self, client: TestClient
    ) -> None:
        # The loader raises InvalidJTLError for this too, so it shares the
        # invalid_jtl code and is distinguished only by the message.
        response = client.post(
            "/analyze", files=_upload(), data={"warmup_seconds": "99999"}
        )
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_jtl"

    def test_analyze_negative_warmup_returns_422(self, client: TestClient) -> None:
        response = client.post("/analyze", files=_upload(), data={"warmup_seconds": "-5"})
        assert response.status_code == 422


class TestGranularEndpoints:
    OTHER_AGENT_KEYS = ("stats", "errors", "anomalies", "trends")

    def _assert_unwrapped(self, body: dict) -> None:
        """Assert the body is a bare report, not a full AnalysisResult."""
        for key in self.OTHER_AGENT_KEYS:
            assert key not in body, f"granular response should not nest '{key}'"
        assert "dataset_metadata" in body

    def test_statistician_endpoint_returns_stats_only(self, client: TestClient) -> None:
        response = client.post("/analyze/statistician", files=_upload())
        assert response.status_code == 200
        body = response.json()
        self._assert_unwrapped(body)
        assert body["total_count"] == 6
        assert "global_p95_ms" in body
        assert {f["name"] for f in body["per_feature"]} == {"Search", "Checkout"}

    def test_errors_endpoint_returns_errors_only(self, client: TestClient) -> None:
        response = client.post("/analyze/errors", files=_upload())
        assert response.status_code == 200
        body = response.json()
        self._assert_unwrapped(body)
        assert set(body["codes_by_feature"]) == {"Search", "Checkout"}
        codes = {b["code"] for b in body["codes_by_feature"]["Checkout"]}
        assert codes == {"200", "500"}

    def test_anomalies_endpoint_returns_anomalies_only(self, client: TestClient) -> None:
        response = client.post("/analyze/anomalies", files=_upload())
        assert response.status_code == 200
        body = response.json()
        self._assert_unwrapped(body)
        # 3 samples per feature is below the 20-sample detection floor.
        assert body["by_feature"]["Search"]["insufficient_data"] is True

    def test_trends_endpoint_returns_trends_only(self, client: TestClient) -> None:
        response = client.post("/analyze/trends", files=_upload())
        assert response.status_code == 200
        body = response.json()
        self._assert_unwrapped(body)
        assert body["by_feature"]["Search"]["windows"] == []

    @pytest.mark.parametrize(
        "endpoint", ["statistician", "errors", "anomalies", "trends"]
    )
    def test_granular_endpoints_reject_invalid_file(
        self, client: TestClient, endpoint: str
    ) -> None:
        response = client.post(
            f"/analyze/{endpoint}", files={"file": ("junk.jtl", GARBAGE_BYTES)}
        )
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_jtl"

    @pytest.mark.parametrize(
        "endpoint", ["statistician", "errors", "anomalies", "trends"]
    )
    def test_granular_endpoints_accept_warmup(
        self, client: TestClient, endpoint: str
    ) -> None:
        response = client.post(
            f"/analyze/{endpoint}", files=_upload(), data={"warmup_seconds": "1"}
        )
        assert response.status_code == 200
        assert response.json()["dataset_metadata"]["warmup_seconds"] == 1.0

    def test_granular_output_matches_composite(self, client: TestClient) -> None:
        # The two paths call the same agent, so results must be identical.
        composite = client.post("/analyze", files=_upload()).json()["stats"]
        granular = client.post("/analyze/statistician", files=_upload()).json()
        assert composite == granular


class TestUploadSizeLimit:
    """Both enforcement layers.

    ``_api_config`` is patched rather than overridden via
    ``app.dependency_overrides`` because the middleware calls
    ``get_api_config()`` directly; patching the module attribute covers the
    middleware and the dependency together.
    """

    @pytest.fixture
    def tiny_limit_client(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
        monkeypatch.setattr(
            dependencies,
            "_api_config",
            ApiConfig(host="127.0.0.1", port=8000, max_upload_mb=0.001),
        )
        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client

    def test_oversized_content_length_returns_413(
        self, tiny_limit_client: TestClient
    ) -> None:
        response = tiny_limit_client.post(
            "/analyze", files={"file": ("big.jtl", b"x" * 5000)}
        )
        assert response.status_code == 413
        assert response.json()["error"] == "file_too_large"

    def test_oversized_chunked_upload_returns_413(
        self, tiny_limit_client: TestClient
    ) -> None:
        # Chunked transfer encoding omits Content-Length, bypassing the
        # middleware. The spooling byte counter is the layer that catches this;
        # without it the cap would be trivially evadable.
        boundary = "----jtlboundary"
        payload = b"x" * 5000
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="big.jtl"\r\n\r\n'
        ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()

        def chunks() -> Iterator[bytes]:
            for start in range(0, len(body), 512):
                yield body[start : start + 512]

        response = tiny_limit_client.post(
            "/analyze",
            content=chunks(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        assert "content-length" not in {k.lower() for k in response.request.headers}
        assert response.status_code == 413
        assert response.json()["error"] == "file_too_large"

    def test_within_limit_upload_succeeds(self, client: TestClient) -> None:
        # Default 200 MB limit leaves the fixtures well clear of the cap.
        assert client.post("/analyze", files=_upload()).status_code == 200


class TestErrorHandling:
    def test_unexpected_exception_returns_500(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jtl_analyzer.api import main

        def boom(dataset: object) -> None:
            raise RuntimeError("secret internal detail: /srv/private/path.py line 42")

        monkeypatch.setattr(main.statistician_module, "run", boom)

        response = client.post("/analyze", files=_upload())
        assert response.status_code == 500
        assert response.json() == {
            "error": "internal_error",
            "message": "An unexpected error occurred",
        }

    def test_500_response_leaks_no_internals(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jtl_analyzer.api import main

        secret = "secret internal detail: /srv/private/path.py line 42"

        def boom(dataset: object) -> None:
            raise RuntimeError(secret)

        monkeypatch.setattr(main.trends_module, "run", boom)

        response = client.post("/analyze", files=_upload())
        assert response.status_code == 500
        text = response.text
        assert secret not in text
        assert "Traceback" not in text
        assert "RuntimeError" not in text

    def test_temp_file_removed_after_failure(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The yield-dependency's finally block must run even when the handler
        # raises, or the API would leak a temp file per failed request.
        from jtl_analyzer.api import main

        spooled: list[str] = []
        original_spool = dependencies._spool_to_temp_file

        def spy(upload, config):  # type: ignore[no-untyped-def]
            path = original_spool(upload, config)
            spooled.append(path)
            return path

        def boom(dataset: object) -> None:
            raise RuntimeError("failure after the temp file existed")

        monkeypatch.setattr(dependencies, "_spool_to_temp_file", spy)
        monkeypatch.setattr(main.statistician_module, "run", boom)

        assert client.post("/analyze", files=_upload()).status_code == 500
        assert spooled, "nothing was spooled, so the test proves nothing"
        assert not Path(spooled[0]).exists()


class TestDocumentation:
    def test_openapi_schema_is_served(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        paths = response.json()["paths"]
        assert set(paths) >= {
            "/health",
            "/analyze",
            "/analyze/statistician",
            "/analyze/errors",
            "/analyze/anomalies",
            "/analyze/trends",
        }

    def test_endpoints_carry_descriptions(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        for path in ("/analyze", "/analyze/statistician", "/analyze/trends"):
            assert schema["paths"][path]["post"]["description"].strip()

    @pytest.mark.parametrize("path", ["/docs", "/redoc"])
    def test_interactive_docs_are_served(self, client: TestClient, path: str) -> None:
        assert client.get(path).status_code == 200
