# Latent Frontier Lab

Interactive educational explainer for the **DataForge 2026: Pathway Track — Explain the Frontier** brief.

## Central lesson

> **A fixed-size latent state can be refined repeatedly to infer a task rule from demonstrations and solve a new query without emitting a verbal chain-of-thought — but more recurrence is not automatically better when evidence is ambiguous or noisy.**

The experience makes this claim falsifiable: learners can change demonstration count, noise, recurrence steps, update strength, and readout temperature, then compare the predicted grid with ground truth.

## What is live vs. precomputed

- **Live:** the toy latent-rule solver, recurrent state trajectory, hypothesis probabilities, predictions, and toy synaptic-memory matrix are computed in the browser session.
- **Synthetic:** the ARC-like grids and hypothesis space are generated locally from a fixed seed.
- **Not official BDH:** `core/bdh_bridge.py` visualizes the published BDH synaptic-state equations. It does not load or reproduce an official BDH/BDH-CQ checkpoint.
- **No scripted animation:** charts reflect the computed state at each recurrence step.

## Learning journey

1. **Observe:** inspect demonstrations and a new query.
2. **Interact:** change a meaningful variable.
3. **See state:** watch a fixed-size latent vector refine over recurrence.
4. **Truth beside estimate:** compare the predicted grid against ground truth.
5. **Challenge:** add noise or remove demonstrations and test the limits.
6. **Bridge to BDH/BDH-CQ:** see how recurrent memory and latent reasoning connect to the BDH framing.

## Architecture

- `app.py` — Streamlit visual essay and controls.
- `core/latent_reasoning.py` — original toy recurrent hypothesis reasoner.
- `core/bdh_bridge.py` — original educational implementation of the simplified synaptic-memory visualization.
- `tests/` — deterministic correctness checks.
- `concept_summary.md` — one-page concept summary source.
- `references.md` — primary research references and evidence labels.

## Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The Docker option is:

```bash
docker build -t latent-frontier-lab .
docker run -p 8501:8501 latent-frontier-lab
```

## Reproducibility

The synthetic puzzle generator and noisy perturbations are seeded. Run the tests with:

```bash
python -m unittest discover -s tests -v
```

## Research grounding

The artifact is designed around the brief's requirements for one precise claim, a real computational substrate, visible state, truth beside estimate, fast feedback, a substantive BDH/BDH-CQ connection, and explicit labeling of toy/precomputed content.

Primary technical references are listed in `references.md`.

## Limitations and honesty

This project intentionally uses a **small hypothesis space** instead of a full neural model. That makes the state inspectable and keeps feedback under a second, but it does not demonstrate that a real language model will behave identically. The toy solver can also become overconfident under ambiguous/noisy demonstrations because its hypothesis space is finite and deliberately simplified. The BDH section is a mechanistic bridge, not a benchmark reproduction.

## AI assistance / provenance disclosure

AI assistance was used to accelerate scaffolding, code generation, copy editing, and packaging. The team submitting this artifact should review, understand, test, and defend every component, every equation, and every research claim. Any external code, assets, data, or fonts should be listed here before submission. No external assets are required by the default build.

## Submission checklist

Before submitting, add:

- public artifact URL
- public source repository URL
- team name / authors
- mentor disclosure, if applicable
- final one-page concept summary PDF
- confirmed licenses and any reused code/assets
- screenshots or demo video, if desired

