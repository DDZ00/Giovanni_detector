"""End-to-end tests for the /analyze HTTP route.

No real model is loaded: `scorers_module.get_scorers` is monkey-patched to
return a pair of `_FakeScorer`s producing deterministic logprobs. This proves
the full request → auth → input-validation → orchestration → JSON-projection
loop works without any GGUF files on disk.
"""
import json

import numpy as np
import pytest


TOKEN = "testtoken"
LONG_TEXT = " ".join(f"word{i}" for i in range(80))  # > Config.MIN_WORDS (50)


class _FakeScorer:
    """Returns a frozen logprob array; ignores text/stride/max_chunk."""

    def __init__(self, seed: int):
        rng = np.random.RandomState(seed)
        # 200 logprobs, in the realistic range roughly (-10, 0)
        self._lp = (rng.rand(200).astype(np.float32) * 8.0 - 8.0)

    def logprobs(self, text, *, stride=10, max_chunk=2048):
        return self._lp


@pytest.fixture
def client(monkeypatch):
    # Set the auth token before any request runs. Class attribute is read
    # fresh on each request, so monkeypatch is sufficient (no app re-import).
    monkeypatch.setattr("config.Config.AUTH_TOKEN", TOKEN)
    # Inject fake scorers so get_scorers never tries to build_scorer_pair().
    monkeypatch.setattr(
        "modules.scorers.get_scorers",
        lambda *a, **kw: (_FakeScorer(seed=1), _FakeScorer(seed=2)),
    )
    # The route imports `scorers_module` and calls .get_scorers() on it, so
    # also patch the bound name app.py used at import time.
    monkeypatch.setattr(
        "app.scorers_module.get_scorers",
        lambda *a, **kw: (_FakeScorer(seed=1), _FakeScorer(seed=2)),
    )
    from app import app as flask_app
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def _auth():
    return {"X-Giovanni-Token": TOKEN}


# Auth was removed (see docs/ROADMAP.md 2026-04-30: "Token auth removed").
# LAN-only deployment, single user. nginx limit_req remains the abuse gate.

# ── Input validation ────────────────────────────────────────────
class TestAnalyzeInputValidation:
    def test_non_json_content_type_is_400(self, client):
        r = client.post("/analyze", data="text=hello", headers=_auth())
        assert r.status_code == 400
        assert "Content-Type" in r.get_json()["error"]

    def test_missing_text_field_is_400(self, client):
        r = client.post("/analyze", json={}, headers=_auth())
        assert r.status_code == 400

    def test_empty_text_is_400(self, client):
        r = client.post("/analyze", json={"text": "   "}, headers=_auth())
        assert r.status_code == 400

    def test_text_must_be_string_not_int(self, client):
        r = client.post("/analyze", json={"text": 12345}, headers=_auth())
        assert r.status_code == 400


# ── Behaviour ───────────────────────────────────────────────────
class TestAnalyzeBehaviour:
    def test_short_text_returns_200_with_error_key(self, client):
        # Below MIN_WORDS — analyzer returns {"error": "...", "categoria": None}
        # inline. Transport-wise this is success; the client renders the error.
        r = client.post("/analyze", json={"text": "only three words"}, headers=_auth())
        assert r.status_code == 200
        body = r.get_json()
        assert "error" in body
        assert body["categoria"] is None

    def test_happy_path_returns_full_metrics(self, client):
        r = client.post("/analyze", json={"text": LONG_TEXT}, headers=_auth())
        assert r.status_code == 200
        body = r.get_json()
        # Required keys from analyze_text + diagnose
        for key in (
            "categoria", "fastdetect_score", "roughness_raw",
            "roughness_normalized", "shadow_score", "thresholds",
            "palabras", "tokens_analizados", "n_ventanas", "diagnostico",
        ):
            assert key in body, f"missing key: {key}"
        assert body["palabras"] == 80
        assert isinstance(body["diagnostico"], str)

    def test_response_strips_private_keys(self, client):
        # _logprobs1, _logprobs2, _roughness_ventanas are numpy/tuple — not
        # JSON-serializable. The route MUST filter them out.
        r = client.post("/analyze", json={"text": LONG_TEXT}, headers=_auth())
        assert r.status_code == 200
        body = r.get_json()
        for key in body.keys():
            assert not key.startswith("_"), f"private key leaked: {key}"

    def test_verbose_flag_includes_logprobs(self, client):
        r = client.post(
            "/analyze",
            json={"text": LONG_TEXT, "verbose": True},
            headers=_auth(),
        )
        assert r.status_code == 200
        body = r.get_json()
        assert "verbose_logprobs1" in body
        assert "verbose_logprobs2" in body
        assert "verbose_ventanas" in body
        assert isinstance(body["verbose_logprobs1"], list)


# ── Failure modes ───────────────────────────────────────────────
class TestAnalyzeFailureModes:
    def test_model_load_failure_is_503(self, client, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("GGUF not found")
        monkeypatch.setattr("app.scorers_module.get_scorers", boom)
        r = client.post("/analyze", json={"text": LONG_TEXT}, headers=_auth())
        assert r.status_code == 503
        body = r.get_json()
        assert body["error"] == "model unavailable"
        assert "GGUF not found" in body["detail"]


# ── /ping stays public, /analyze stays protected ────────────────
class TestRouteVisibility:
    def test_ping_is_public(self, client):
        r = client.get("/ping")
        assert r.status_code == 200

    def test_health_is_public(self, client):
        # Auth removed (LAN-only). /health returns version + db counts.
        r = client.get("/health")
        assert r.status_code == 200
        body = r.get_json()
        assert body["status"] == "healthy"
        assert "counts" in body and "samples" in body["counts"]
