"""
Giovanni configuration.

Detection thresholds are lifted verbatim from the original Colab notebook
(Giovanni-HumanityDetector.ipynb, cell 2). They will need recalibration
once we move from bitsandbytes NF4 (notebook) to GGUF Q4_K_M (production)
quantisation — the v1 calibration UI exists for that.
"""
import os


class Config:
    # ── Service identity ────────────────────────────────────────
    VERSION = "0.3.0"
    SERVICE_NAME = "giovanni"

    # ── Server ──────────────────────────────────────────────────
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 5004
    APPLICATION_ROOT = os.getenv("APPLICATION_ROOT", "/giovanni")
    DEBUG = os.getenv("FLASK_ENV") != "production"

    # ── Auth ────────────────────────────────────────────────────
    AUTH_TOKEN = os.getenv("GIOVANNI_AUTH_TOKEN", "")

    # ── Binoculars detector (Hans et al. 2024) ──────────────────
    # B = log PPL_M1 / log X-PPL_M1,M2.  Below threshold → machine-generated.
    # Paper's global threshold is 0.901; OOD-tuned 0.9.
    #
    # RANGE:    typical [0.5, 1.2].
    # POLARITY: B < threshold → AI;   B > threshold → human.
    # PAIR:     Falcon-7B base + Falcon-7B-Instruct — the same pair the
    #           Binoculars paper used. The threshold below is the paper's
    #           value, applied verbatim. This is the load-bearing detector
    #           signal; the calculus blend below is a secondary check.
    THRESHOLD_BINOCULARS = 0.901

    # ── Calculus strategy (the personal contribution) ───────────
    # Total-variation of the base-model log-prob first derivative.
    #
    # RANGE:    typical [200, 3000] for ~500-word texts.
    # POLARITY: tv2 > threshold → AI;   tv2 < threshold → human.
    #           NOTE: this is the OPPOSITE of what the calibration plan
    #           (docs/plans/GIOVANNI_SCORE_CALIBRATION.md) documented as
    #           the expected case. The threshold below was set empirically
    #           against a small handful of samples, which polarized this way.
    # OBSERVED (small corpus, 2026-05-02; tv2 uses base model only):
    #     human-l1=1685.3, ai-clean=1876.9, ai-clean=1867.6
    # NOTE: tv2 magnitudes are model-pair-specific. These observations were
    # taken on a different scorer pair than the current Falcon-only setup;
    # the threshold may need adjustment once Falcon tv2 is observed in
    # practice. Treat verdicts as interpretive, not statistical.
    THRESHOLD_TV2 = 1775.0

    # Master switch. False = pure Binoculars verdict (paper baseline).
    # True = Binoculars soft-combined with calculus tv2 via sigmoid blend.
    USE_CALCULUS = True

    # Soft-combine knobs. p_ai = w * p_bino + (1-w) * p_calc.
    #
    # BINOCULARS_WEIGHT
    #   RANGE:    [0.0, 1.0].
    #   MEANING:  fraction of the final verdict driven by Binoculars.
    #             0.0 = calculus-only;  1.0 = Binoculars-only.
    #   VALUE:    0.7 — Falcon paper baseline. Binoculars is the load-
    #             bearing signal on the Falcon pair; calculus tv2 is the
    #             secondary check. If the dashboard's verdicts feel
    #             dominated by tv2 idiosyncrasies, raise toward 1.0;
    #             if tv2 looks more discriminating in practice, lower.
    BINOCULARS_WEIGHT = 0.7

    # BINOCULARS_STEEPNESS
    #   RANGE:    [1, 30]; sigmoid sharpness for p_bino.
    #             p_bino = sigmoid((threshold - B) * k).
    #   READING:  higher k → more decisive votes (closer to 0 or 1);
    #             lower k → more borderline values near 0.5.
    #   RETUNE:   only matters when BINOCULARS_WEIGHT > 0. Tune by
    #             watching the p_bino histogram on the corpus: if it
    #             clusters near 0.5, raise k; if it pegs at 0/1, lower.
    BINOCULARS_STEEPNESS = 10.0

    # CALCULUS_STEEPNESS
    #   RANGE:    [1, 30]; sigmoid sharpness for p_calc.
    #             p_calc = sigmoid((tv2/threshold - 1) * k).
    #   VALUE:    10.0 — produces decisive p_calc verdicts (≥0.12 margin
    #             from 0.5) on the small calibration corpus. Lower toward
    #             4.0 if borderline values feel undervalued; raise toward
    #             20.0 if the calculus signal is rarely uncertain.
    CALCULUS_STEEPNESS = 10.0

    # ── REMINDER ─────────────────────────────────────────────────
    # Giovanni is a working interpretive tool, not a statistically validated
    # classifier. THRESHOLD_TV2 and CALCULUS_STEEPNESS were picked from a
    # small corpus (2026-05-02) and refined empirically. THRESHOLD_BINOCULARS
    # is the Hans et al. 2024 paper value, applied verbatim on the pair the
    # paper used. Treat individual verdicts as one signal among many; if you
    # disagree with a verdict, the dashboard exposes the underlying numbers
    # so you can see why.
    # ─────────────────────────────────────────────────────────────

    # ── Processing parameters ───────────────────────────────────
    MIN_WORDS = 50
    STRIDE = 10
    MAX_CHUNK_TOKENS = 2048

    # ── Models (Giovanni-owned GGUF files, no Ollama coupling) ──
    # Place GGUF files in MODELS_DIR; default filenames match scripts/fetch_models.sh.
    MODELS_DIR = os.getenv("GIOVANNI_MODELS_DIR", "/models")
    MODEL_BASE_PATH = os.getenv(
        "GIOVANNI_MODEL_BASE_PATH",
        os.path.join(MODELS_DIR, "falcon-7b-q4_k_m.gguf"),
    )
    MODEL_INSTRUCT_PATH = os.getenv(
        "GIOVANNI_MODEL_INSTRUCT_PATH",
        os.path.join(MODELS_DIR, "falcon-7b-instruct-q4_k_m.gguf"),
    )

    # ── Model pair ─────────────────────────────────────────────
    # Falcon-7B base + Falcon-7B-Instruct — the pair the original Binoculars
    # paper (Hans et al. 2024) used. `same_tokenizer=True` is required for
    # Binoculars (X-PPL needs aligned distributions).
    MODEL_PAIRS = {
        "falcon": {
            "label": "Binoculars-original — Falcon-7B base + Falcon-7B-Instruct",
            "base": os.path.join(MODELS_DIR, "falcon-7b-q4_k_m.gguf"),
            "instruct": os.path.join(MODELS_DIR, "falcon-7b-instruct-q4_k_m.gguf"),
            "same_tokenizer": True,
        },
    }
    DEFAULT_PAIR_ID = os.getenv("GIOVANNI_DEFAULT_PAIR", "falcon")

    # ── Paths ───────────────────────────────────────────────────
    DATA_DIR = os.getenv("GIOVANNI_DATA_DIR", "/app/data")
    REPORTS_DIR = os.getenv("GIOVANNI_REPORTS_DIR", "/app/data/reports")
    UPLOADS_DIR = os.getenv("GIOVANNI_UPLOADS_DIR", "/app/data/uploads")
    DB_PATH = os.getenv("GIOVANNI_DB_PATH", "/app/data/giovanni.db")

    # ── Inputs ──────────────────────────────────────────────────
    ALLOWED_EXTENSIONS = {"txt", "docx"}
    MAX_UPLOAD_SIZE_MB = 10
