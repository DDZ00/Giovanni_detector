# Giovanni

An interpretive research tool for detecting AI-generated text. Combines the **Binoculars** detector (Hans et al., ICML 2024) with a complementary **calculus signal** (total variation of base-model log-probability first derivative).

## Purpose

Giovanni is a **non-profit research project**. All code and configuration in this repository is provided for **academic and educational research purposes**. It is a small-corpus interpretive tool — verdicts should be treated as *one signal among many*, not as authoritative classifications. It is **not** a statistically validated classifier and should not be used as the sole basis for decisions about authorship.

## Status

Working interpretive tool. Calibrated against a small corpus on a single model pair (Falcon-7B base + Falcon-7B-Instruct). The Binoculars threshold (`0.901`) is the Hans et al. (2024) paper value, applied verbatim on the same Falcon-7B pair the paper used. The calculus threshold (`tv2 = 1775.0`) was set empirically against a handful of samples and is documented inline as such — see `config.py`.

## How it works

- **Binoculars** (`B = log PPL_M1 / log X-PPL_M1,M2`): paper-baseline detector. Below threshold → likely machine-generated; above threshold → likely human.
- **Calculus signal** (`tv2 = Σ|l(t+1) − l(t)|`): total variation of the base model's log-probability first derivative. Secondary check, soft-combined with Binoculars via a sigmoid blend (`p_ai = w·p_bino + (1−w)·p_calc`).

The web dashboard exposes both signals plus the underlying numbers, so a researcher can see *why* Giovanni reached a verdict — and disagree with it.

## Run (Docker)

```bash
# Fetch GGUF model weights — they are not in this repo (multi-GB, not redistributable)
bash scripts/fetch_models.sh
GIOVANNI_FETCH_FALCON=1 bash scripts/fetch_models.sh   # Falcon pair (~9 GB)

# Build and launch
docker compose build giovanni-app
docker compose up -d giovanni-app
```

Then open the dashboard at the configured route (default `http://localhost:5004/giovanni`).

## Repository layout

```
app.py              Flask routes (samples, batch upload, analyze, runs)
config.py           Thresholds, model paths, processing parameters
modules/            Binoculars, calculus, scorers, storage, PDF parser
templates/          Dashboard HTML (single-page)
static/             CSS
scripts/            fetch_models.sh, export_corpus.py
tests/              pytest suite (parser, batch upload, storage migrations)
```

## Citations

If you use Giovanni in academic work, please cite the upstream Binoculars paper:

- Hans, A., Schwarzschild, A., Cherepanova, V., Kazemi, H., Saha, A., Goldblum, M., Geiping, J., & Goldstein, T. (2024). *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text*. ICML 2024.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

This repository is a non-profit research artifact. The Apache 2.0 license permits commercial use, but **the intended use is academic and educational research**. Please honor that spirit.
