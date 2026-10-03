# Primary research references

## Selected papers

1. **Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., & Tian, Y. (2024/2025). _Training Large Language Models to Reason in a Continuous Latent Space._** arXiv:2412.06769; COLM 2025.
   - Core relevance: Coconut replaces selected verbal reasoning steps with continuous hidden-state “thoughts”.
   - https://arxiv.org/abs/2412.06769

2. **Kosowski, A., Uznański, P., Chorowski, J., Stamirowska, Z., & Bartoszkiewicz, M. (2025). _The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain._** arXiv:2509.26507.
   - Core relevance: BDH, sparse positive activations, Hebbian working memory, graph/synapse interpretation, GPU-friendly formulation.
   - https://arxiv.org/abs/2509.26507

3. **Wang, G., Li, J., Sun, Y., Chen, X., Liu, C., Wu, Y., Lu, M., Song, S., & Abbasi-Yadkori, Y. (2025). _Hierarchical Reasoning Model._** arXiv:2506.21734.
   - Core relevance: recurrent hidden-state reasoning with multiple timescales and no explicit chain-of-thought supervision.
   - https://arxiv.org/abs/2506.21734

4. **Engdahl, B., Kosowski, A., Chorowski, J., Stamirowska, Z., Uznański, P., Jiang, J., Phadke, R., Kinas, R., & Zhong, R. (2026). _BDH-CQ: In-Context Learning with Recurrent Latent Reasoning._** arXiv:2608.09888.
   - Core relevance: inference-time demonstrations update recurrent memory; the model then solves a query through iterative latent computation without verbalizing intermediate reasoning.
   - https://arxiv.org/abs/2608.09888

## BDH derivation reference

Pathway Research, **“From attention to synapses: deriving BDH”** (2026). The explainer shows the matrix-form construction:

`σ_t = σ_{t-1} + x_t^T v_t`

`o_t = x_t σ_t`

and describes the outer-product write as Hebbian-like synaptic strengthening.

https://pathway.com/research/bdh-explainer/bdh-architecture-derivation

## Evidence discipline

- Paper claims are attributed to the cited primary sources.
- The grid solver is an original toy educational system.
- The BDH matrix panel is an educational visualization of the published derivation, not an official BDH implementation.
- No benchmark numbers in the app are presented as results produced by this toy project.
