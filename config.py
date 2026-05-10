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
    VERSION = "0.3.2"
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
    #           Binoculars paper used. Native-English (or unknown
    #           proficiency) verdicts use this threshold, the paper value
    #           applied verbatim.
    THRESHOLD_BINOCULARS = 0.901

    # Non-native English writers produce systematically lower B because their
    # token sequences are smoother / more formulaic to a Falcon-7B base
    # trained on English-dominated web corpora — a known Binoculars
    # limitation flagged in Hans et al. §6.2. Calibrated 2026-05-09 against
    # 7 IB Extended Essays (English B SL) from 2018, all known-human (the
    # cohort predates student-accessible AI):
    #     B observed range 0.826 – 0.885 (mean 0.86, σ ≈ 0.02)
    #     min margin above 0.82 = 0.006 — thin; widen the corpus before
    #     tightening this number further.
    # Engaged when analyze_text() receives english_proficiency="non_native".
    # Trade-off: AI samples in the same B band (Gemini=0.862, DeepSeek=0.882
    # in the small calibration corpus) will also score Human under this
    # threshold. Falcon-7B Binoculars cannot resolve that overlap; only a
    # different model pair or a real labelled classifier can.
    THRESHOLD_BINOCULARS_NON_NATIVE = 0.82

    # ── Calculus strategy (deprecated 2026-05-09) ───────────────
    # tv2 = total-variation Σ|l(t+1) − l(t)| of the base-model logprob curve.
    # As an UNNORMALIZED sum, it scales linearly with text length; once you
    # divide by token count the human/AI distributions overlap (humans
    # 3.09–3.97 per token, AI 2.86–3.82 per token on the 11-sample corpus
    # 2026-05-09). The discriminative power that originally tuned
    # THRESHOLD_TV2 was almost entirely a length proxy, not AI-vs-human
    # signal. USE_CALCULUS is now False; the constants below are kept for
    # historical context and for the dashboard's "show me the numbers" view.
    THRESHOLD_TV2 = 1775.0

    # Master switch. False = pure Binoculars verdict (paper baseline +
    # proficiency-aware threshold). True kept available for ablation /
    # research mode but no longer the default.
    USE_CALCULUS = False

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
    # classifier. THRESHOLD_BINOCULARS is the Hans et al. 2024 paper value
    # applied verbatim. THRESHOLD_BINOCULARS_NON_NATIVE was calibrated on a
    # 7-essay non-native English corpus (2018 IB EE) — the corpus is small,
    # and Falcon-7B Binoculars cannot resolve non-native human vs AI inside
    # the [0.82, 0.95] B band on this pair. Treat verdicts in that band as
    # one signal among many; if you disagree with a verdict, the dashboard
    # exposes B, the threshold actually used, and the calculus signals.
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
    MAX_UPLOAD_SIZE_MB = int(os.getenv("GIOVANNI_MAX_UPLOAD_MB", "200"))
