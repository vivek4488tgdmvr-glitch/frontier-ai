# Latent Reasoning, Made Visible

## Central claim

A fixed-size latent state can be refined repeatedly to infer a task rule from demonstrations and solve a new query without emitting a verbal chain-of-thought—but extra recurrence cannot compensate for missing or contradictory evidence.

## Why this matters now

A growing research thread asks whether useful reasoning must be serialized into natural-language tokens. Coconut (COLM 2025) studies continuous hidden-state “thoughts”: instead of decoding every intermediate step, selected reasoning states are fed back into the model as continuous representations. HRM (2025) studies recurrent latent reasoning with interacting timescales. BDH (2025) connects local neuron/synapse dynamics with a transformer-like sequence model and treats working memory as synaptic plasticity. BDH-CQ (2026) combines in-context learning with recurrent latent reasoning: inference-time inputs update recurrent memory, after which the model solves a query by iterative computation in latent space without verbalizing intermediate reasoning.

## What this artifact teaches

Latent Frontier Lab turns that abstract design idea into a controlled, inspectable experiment. The live toy system has a finite set of visual transformations (rotation, reflection, shifting, and color maps). Each demonstration provides evidence for every candidate rule. That evidence is accumulated into a fixed-size latent vector of hypothesis logits. Recurrent updates repeatedly refine the same vector; the final state is read out as a probability distribution and used to transform a new query.

The learner can change the number of demonstrations, demonstration corruption, update strength, recurrence steps, and readout temperature. The app shows the demonstrations, the query, the ground-truth answer, the model prediction, and the full trajectory of hypothesis beliefs. This makes the claim falsifiable in under a minute: a learner can increase recurrence, add noise, and observe whether confidence and correctness improve together or diverge.

## BDH connection

The BDH bridge is intentionally concrete and explicitly labeled as a toy visualization. Pathway’s published derivation rewrites attention as a fixed high-dimensional matrix interpreted as synaptic connectivity. With neuron-like activity vectors `x_t` and value vectors `v_t`, the synaptic state is updated by an outer product:

`σ_t = σ_{t-1} + x_t^T v_t`,  and read as `o_t = x_t σ_t`.

The important pedagogical connection is not that the toy solver is BDH; it is that both stories make **state** computationally meaningful. In the toy reasoner, the recurrent state is a compact belief over task rules. In BDH, the published architecture gives a synaptic state a graph-like interpretation and describes Hebbian-like memory updates. BDH-CQ extends the broader story to inference-time context updates plus recurrent latent reasoning.

## Comparison

| Approach | Where reasoning happens | Main trade-off |
|---|---|---|
| Transformer + CoT | Natural-language tokens | Transparent but potentially long/expensive traces |
| Coconut | Continuous hidden state | Less verbalization; latent computation is less directly observable |
| HRM | Recurrent hidden state at multiple timescales | Deeper iterative computation without explicit CoT |
| BDH / BDH-CQ | State + memory + recurrent latent computation | Richer memory story, but still a young research direction |

## Limitation

The artifact is intentionally not a neural checkpoint and does not reproduce any research benchmark. Its hypothesis space is tiny and finite, so it is excellent for visualizing belief refinement but insufficient for conclusions about general intelligence. Under ambiguous demonstrations, the toy system may become confidently wrong because the evidence is compressed into a fixed hypothesis set. That failure is part of the lesson: recurrent compute can refine evidence, but it does not create evidence that is absent from the context.

## Sources

Hao et al. (2024/2025), *Training Large Language Models to Reason in a Continuous Latent Space*, arXiv:2412.06769.

Kosowski et al. (2025), *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain*, arXiv:2509.26507.

Wang et al. (2025), *Hierarchical Reasoning Model*, arXiv:2506.21734.

Engdahl et al. (2026), *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning*, arXiv:2608.09888.
