"""
Giovanni — Research Dashboard (v0.3)
Flask application. All routes public (LAN-only deployment).
"""
import io
import json
import logging
import os
import time
import zipfile

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.middleware.proxy_fix import ProxyFix

from config import Config
from modules import scorers as scorers_module
from modules import storage
from modules.analyzer import analyze_text
from modules.docx_parse import extract_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.config["APPLICATION_ROOT"] = Config.APPLICATION_ROOT
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_UPLOAD_SIZE_MB * 1024 * 1024

GIT_SHA = os.getenv("GIOVANNI_GIT_SHA", "unknown")


@app.context_processor
def inject_version():
    return {"giovanni_version": Config.VERSION}


# ── Health ─────────────────────────────────────────────────────
@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "service": Config.SERVICE_NAME}), 200


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": Config.SERVICE_NAME,
        "version": Config.VERSION,
        "git_sha": GIT_SHA,
        "counts": storage.counts(),
        "model_pairs": scorers_module.list_pairs(),
        "default_pair": Config.DEFAULT_PAIR_ID,
    }), 200


# ── Legacy ad-hoc analyze (paste textarea, no persistence) ─────
@app.route("/analyze", methods=["POST"])
def analyze():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    body = request.get_json(silent=True) or {}
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required and must be a non-empty string"}), 400
    verbose = bool(body.get("verbose", False))
    params = _params_from_dict(body.get("params") or {})
    refusal = _refuse_if_binoculars_on_cross_pair(params)
    if refusal:
        return refusal
    logger.info("analyze adhoc: words=%d sha256=%s pair=%s n_ctx=%d use_calculus=%s",
                len(text.split()), _short_sha(text), params["pair_id"], params["n_ctx"], params["use_calculus"])

    try:
        scorer_base, scorer_instruct = scorers_module.get_scorers(pair_id=params["pair_id"], n_ctx=params["n_ctx"])
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to load scorer pair")
        return jsonify({"error": "model unavailable", "detail": str(e)}), 503

    t0 = time.perf_counter()
    try:
        result = analyze_text(text, scorer_base, scorer_instruct, verbose=verbose, **_analyzer_kwargs(params))
    except FileNotFoundError as e:
        logger.exception("analyze adhoc failed: missing model file")
        return jsonify({"error": "model file missing", "detail": str(e),
                        "remedy": "fetch the GGUF or pick a different pair_id"}), 503
    dt = time.perf_counter() - t0
    public = {k: v for k, v in result.items() if not k.startswith("_")}
    logger.info("analyze adhoc done: %.2fs categoria=%s B=%s tv2=%s p_ai=%s",
                dt, public.get("categoria"),
                (public.get("binoculars") or {}).get("score"),
                (public.get("differential") or {}).get("tv2"),
                (public.get("probabilities") or {}).get("p_ai"))
    return jsonify(public), 200


# ── Samples ────────────────────────────────────────────────────
@app.route("/samples", methods=["POST"])
def create_sample():
    label = english_proficiency = source = filename = notes = None
    text = None
    if request.content_type and request.content_type.startswith("multipart/form-data"):
        f = request.files.get("file")
        label = (request.form.get("label") or "").strip() or None
        english_proficiency = (request.form.get("english_proficiency") or "").strip() or None
        source = (request.form.get("source") or "").strip() or None
        notes = (request.form.get("notes") or "").strip() or None
        if f:
            filename = f.filename
            try:
                text = extract_text(filename or "", f.read())
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
        else:
            text = (request.form.get("text") or "").strip() or None
    else:
        body = request.get_json(silent=True) or {}
        text = body.get("text")
        label = body.get("label")
        english_proficiency = body.get("english_proficiency")
        source = body.get("source")
        notes = body.get("notes")
        filename = body.get("filename")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "text or file required"}), 400
    try:
        rec = storage.add_sample(
            text, label=label, english_proficiency=english_proficiency,
            source=source, filename=filename, notes=notes,
        )
    except storage.DuplicateSample as e:
        return jsonify({"error": str(e)}), 409
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(rec), 201


@app.route("/samples/batch", methods=["POST"])
def create_samples_batch():
    """Accept multiple files (e.g. a folder of PDFs) in one request.

    Each file becomes its own sample row. Per-file outcomes are returned
    so the UI can show duplicates / parse errors without aborting the batch.
    """
    if not (request.content_type or "").startswith("multipart/form-data"):
        return jsonify({"error": "multipart/form-data required"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "no files provided (use field name 'files')"}), 400

    label = (request.form.get("label") or "").strip() or None
    english_proficiency = (request.form.get("english_proficiency") or "").strip() or None
    source_prefix = (request.form.get("source") or "").strip() or None
    notes = (request.form.get("notes") or "").strip() or None

    results = []
    added = 0
    for f in files:
        filename = f.filename or ""
        entry = {"filename": filename}
        try:
            text = extract_text(filename, f.read())
            src = source_prefix or filename
            rec = storage.add_sample(
                text, label=label, english_proficiency=english_proficiency,
                source=src, filename=filename, notes=notes,
            )
            entry.update(status="added", sample_id=rec["id"], n_words=rec["n_words"])
            added += 1
        except storage.DuplicateSample as e:
            entry.update(status="duplicate", error=str(e))
        except ValueError as e:
            entry.update(status="error", error=str(e))
        except Exception as e:  # noqa: BLE001 — never let one bad file kill the batch
            logger.exception("batch upload: failed on %s", filename)
            entry.update(status="error", error=f"{type(e).__name__}: {e}")
        results.append(entry)

    logger.info("batch upload: %d files, %d added", len(files), added)
    return jsonify({"added": added, "total": len(files), "results": results}), 201


@app.route("/samples", methods=["GET"])
def list_samples_route():
    label = request.args.get("label") or None
    english_proficiency = request.args.get("english_proficiency") or None
    samples = storage.list_samples(label=label, english_proficiency=english_proficiency)
    return jsonify({
        "samples": samples,
        "valid_labels": sorted(storage.VALID_LABELS),
        "valid_proficiency": sorted(storage.VALID_PROFICIENCY),
    }), 200


@app.route("/samples/<int:sample_id>", methods=["GET"])
def get_sample_route(sample_id: int):
    s = storage.get_sample(sample_id)
    if not s:
        return jsonify({"error": "not found"}), 404
    return jsonify(s), 200


@app.route("/samples/<int:sample_id>", methods=["DELETE"])
def delete_sample_route(sample_id: int):
    return ("", 204) if storage.delete_sample(sample_id) else (jsonify({"error": "not found"}), 404)


@app.route("/samples/export", methods=["GET"])
def export_samples_route():
    return Response(stream_with_context(storage.export_samples_jsonl()),
                    mimetype="application/x-jsonlines",
                    headers={"Content-Disposition": 'attachment; filename="samples.jsonl"'})


# ── Runs ───────────────────────────────────────────────────────
@app.route("/samples/<int:sample_id>/analyze", methods=["POST"])
def analyze_sample(sample_id: int):
    s = storage.get_sample(sample_id)
    if not s:
        return jsonify({"error": "sample not found"}), 404

    body = request.get_json(silent=True) or {}
    params = _params_from_dict(body.get("params") or {})
    refusal = _refuse_if_binoculars_on_cross_pair(params)
    if refusal:
        return refusal
    verbose = bool(body.get("verbose", True))  # default verbose for stored runs
    logger.info("analyze sample #%d (%s): words=%d sha256=%s params=%s",
                sample_id, s.get("label"), s["n_words"], (s.get("sha256") or "")[:8], params)

    try:
        scorer_base, scorer_instruct = scorers_module.get_scorers(pair_id=params["pair_id"], n_ctx=params["n_ctx"])
    except Exception as e:  # noqa: BLE001
        logger.exception("scorer load failed")
        return jsonify({"error": "model unavailable", "detail": str(e)}), 503

    t0 = time.perf_counter()
    try:
        result = analyze_text(s["text"], scorer_base, scorer_instruct, verbose=verbose, **_analyzer_kwargs(params))
    except FileNotFoundError as e:
        logger.exception("analyze failed: missing model file")
        return jsonify({"error": "model file missing", "detail": str(e),
                        "remedy": "fetch the GGUF or pick a different pair_id"}), 503
    duration_ms = int((time.perf_counter() - t0) * 1000)
    public = {k: v for k, v in result.items() if not k.startswith("_")}

    rec = storage.add_run(sample_id, params, public, duration_ms=duration_ms)
    logger.info("analyze sample #%d done: %dms run #%d categoria=%s B=%s tv2=%s p_ai=%s",
                sample_id, duration_ms, rec["id"], public.get("categoria"),
                (public.get("binoculars") or {}).get("score"),
                (public.get("differential") or {}).get("tv2"),
                (public.get("probabilities") or {}).get("p_ai"))
    return jsonify({"run_id": rec["id"], "duration_ms": duration_ms, "result": public}), 201


@app.route("/samples/<int:sample_id>/runs", methods=["GET"])
def list_runs_route(sample_id: int):
    if not storage.get_sample(sample_id):
        return jsonify({"error": "sample not found"}), 404
    return jsonify({"runs": storage.list_runs(sample_id)}), 200


@app.route("/runs/<int:run_id>", methods=["GET"])
def get_run_route(run_id: int):
    r = storage.get_run(run_id)
    if not r:
        return jsonify({"error": "not found"}), 404
    return jsonify(r), 200


@app.route("/runs/<int:run_id>", methods=["DELETE"])
def delete_run_route(run_id: int):
    return ("", 204) if storage.delete_run(run_id) else (jsonify({"error": "not found"}), 404)


# ── Corpus view ───────────────────────────────────────────────
@app.route("/corpus", methods=["GET"])
def corpus_view():
    """Each sample's most recent run, flattened for scatter/histogram plots."""
    rows = storage.latest_run_per_sample()
    points = []
    for r in rows:
        res = r["result"] or {}
        diff = res.get("differential") or {}
        bino = res.get("binoculars") or {}
        probs = res.get("probabilities") or {}
        points.append({
            "sample_id": r["sample_id"],
            "run_id": r["run_id"],
            "label": r["label"],
            "n_words": r["n_words"],
            "binoculars": bino.get("score"),
            "log_ppl_m1": bino.get("log_ppl_m1"),
            "log_x_ppl": bino.get("log_x_ppl"),
            "p_bino": probs.get("p_bino"),
            "p_calc": probs.get("p_calc"),
            "p_ai": probs.get("p_ai"),
            **diff,
        })
    return jsonify({"points": points}), 200


# ── Export bundle (Phase 5) ────────────────────────────────────
@app.route("/export/corpus.zip", methods=["GET"])
def export_corpus_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("samples.jsonl", "".join(storage.export_samples_jsonl()))
        zf.writestr("runs.jsonl", "".join(storage.export_runs_jsonl()))
        manifest = {
            "schema_version": 1,
            "exported_at": storage._now(),
            "git_sha": GIT_SHA,
            "version": Config.VERSION,
            "counts": storage.counts(),
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("README.md", _BUNDLE_README)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="giovanni-corpus.zip"'})


_BUNDLE_README = """# Giovanni Corpus Export

- `samples.jsonl` — one sample per line: id, text, label, source, filename, sha256, n_words, notes, created_at.
- `runs.jsonl` — one run per line: id, sample_id, params, result (full analyze() output incl. differential signals), duration_ms, created_at.
- `manifest.json` — schema version, export timestamp, git sha, counts.

Loading in pandas:
    import pandas as pd
    samples = pd.read_json("samples.jsonl", lines=True)
    runs    = pd.read_json("runs.jsonl",    lines=True)
"""


# ── UI ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Param helpers ─────────────────────────────────────────────
def _params_from_dict(d: dict) -> dict:
    """Resolve UI-supplied params with config defaults. All values are JSON-safe."""
    pair_id = d.get("pair_id") or Config.DEFAULT_PAIR_ID
    if pair_id not in Config.MODEL_PAIRS:
        pair_id = Config.DEFAULT_PAIR_ID
    return {
        "threshold_binoculars":  float(d.get("threshold_binoculars",  Config.THRESHOLD_BINOCULARS)),
        "threshold_tv2":         float(d.get("threshold_tv2",         Config.THRESHOLD_TV2)),
        "binoculars_weight":     float(d.get("binoculars_weight",     Config.BINOCULARS_WEIGHT)),
        "binoculars_steepness":  float(d.get("binoculars_steepness",  Config.BINOCULARS_STEEPNESS)),
        "calculus_steepness":    float(d.get("calculus_steepness",    Config.CALCULUS_STEEPNESS)),
        "max_chunk":             int(d.get("max_chunk",               Config.MAX_CHUNK_TOKENS)),
        "n_ctx":                 int(d.get("n_ctx",                   scorers_module.DEFAULT_N_CTX)),
        "pair_id":               pair_id,
        "use_calculus":          bool(d.get("use_calculus",           Config.USE_CALCULUS)),
    }


def _analyzer_kwargs(params: dict) -> dict:
    """Subset of params that analyze_text() consumes (pair_id and n_ctx are scorer-side)."""
    pair_spec = Config.MODEL_PAIRS[params["pair_id"]]
    return {
        "threshold_binoculars":  params["threshold_binoculars"],
        "threshold_tv2":         params["threshold_tv2"],
        "binoculars_weight":     params["binoculars_weight"],
        "binoculars_steepness":  params["binoculars_steepness"],
        "calculus_steepness":    params["calculus_steepness"],
        "max_chunk":             params["max_chunk"],
        "use_calculus":          params["use_calculus"],
        "same_tokenizer":        bool(pair_spec.get("same_tokenizer", False)),
    }


def _refuse_if_binoculars_on_cross_pair(params: dict):
    """Return a (response, status) tuple if the request is impossible, else None.

    Binoculars REQUIRES same_tokenizer pairs. If the user picked a cross-pair
    AND has not switched to calculus-only mode (use_calculus=False with no
    Binoculars), refuse with a clear message.
    """
    pair_spec = Config.MODEL_PAIRS[params["pair_id"]]
    if pair_spec.get("same_tokenizer", False):
        return None
    # Cross-tokenizer pair selected. Pure-Binoculars verdict is impossible.
    # Calculus-only is allowed (it doesn't need aligned distributions).
    if not params["use_calculus"]:
        return (jsonify({
            "error": (
                f"Binoculars requires a same-tokenizer pair. "
                f"Pair {params['pair_id']!r} uses different tokenizers. "
                f"Either switch to a same-tokenizer pair (e.g. 'falcon' or 'qwen') "
                f"or enable use_calculus to fall back to calculus-only verdict."
            ),
            "pair_id": params["pair_id"],
            "remedy": "switch to falcon/qwen or enable use_calculus",
        }), 400)
    return None


def _short_sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


if __name__ == "__main__":
    logger.info("Starting Giovanni v%s on %s:%d", Config.VERSION, Config.SERVER_HOST, Config.SERVER_PORT)
    app.run(host=Config.SERVER_HOST, port=Config.SERVER_PORT, debug=Config.DEBUG)
