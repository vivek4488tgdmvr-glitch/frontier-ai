from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from core.latent_reasoning import RULES, make_demo, corrupt_grid, run_recurrent_reasoner, mismatch
from core.bdh_bridge import toy_synapse_replay

st.set_page_config(page_title="Latent Frontier Lab", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.main > div {padding-top: 1.3rem;}
.block-container {max-width: 1250px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 22px; background: linear-gradient(135deg,#141427 0%,#26204a 50%,#1f3d52 100%); color:#fff; margin-bottom:1.2rem;}
.hero h1 {font-size: 3rem; margin:0 0 .25rem 0;}
.hero p {font-size: 1.05rem; opacity:.9; max-width: 900px;}
.claim {padding: 1rem 1.2rem; border:1px solid #d9d9e6; border-radius: 18px; background:#fbfbfe; font-size:1.1rem;}
.kicker {font-weight:700; letter-spacing:.08em; text-transform:uppercase; font-size:.78rem; opacity:.72;}
.small {font-size:.86rem; opacity:.75;}
.grid {border-collapse:separate; border-spacing:3px; margin:auto;}
.grid td {width:34px; height:34px; text-align:center; border-radius:7px; font-weight:700; font-family:ui-monospace,monospace;}
.t0{background:#f1f2f6}.t1{background:#8fd3ff}.t2{background:#ffcf70}.t3{background:#ff8c8c}.t4{background:#c6a7ff}
.badge {padding:.25rem .55rem;border-radius:999px;background:#eef0f8;font-size:.8rem;display:inline-block;margin-right:.3rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="kicker">DataForge × Pathway · Interactive Explainer</div>
  <h1>Latent Frontier Lab</h1>
  <p><b>Can a small fixed-size state reason without spelling out every intermediate step?</b><br>
  Explore recurrent latent-space reasoning, then open the hood on the BDH / BDH-CQ connection.</p>
</div>
<div class="claim"><b>One-sentence claim.</b> A fixed-size latent state can be refined repeatedly to infer a task rule from demonstrations and solve a new query without emitting a verbal chain-of-thought — but more recurrence is not automatically better because ambiguous evidence can saturate or interfere.</div>
""", unsafe_allow_html=True)


def grid_html(g: np.ndarray) -> str:
    rows=[]
    for row in g:
        cells=[]
        for x in row:
            cells.append(f'<td class="t{int(x)}">{int(x) if int(x) else "·"}</td>')
        rows.append('<tr>'+''.join(cells)+'</tr>')
    return '<table class="grid">'+''.join(rows)+'</table>'


def show_grid(g: np.ndarray, caption: str):
    st.markdown(f"**{caption}**")
    st.markdown(grid_html(g), unsafe_allow_html=True)


def build_case(rule_name: str, demos_n: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    rule = next(r for r in RULES if r.name == rule_name)
    demos=[]
    for _ in range(demos_n):
        x,y=make_demo(rng,rule)
        y=corrupt_grid(y,rng,noise)
        demos.append((x,y))
    q,y_true=make_demo(rng,rule)
    return rule,demos,(q,y_true)

with st.sidebar:
    st.markdown("### Experiment controls")
    preset = st.selectbox("Preset", ["Guided: Rotate 90°", "Hard: noisy evidence", "Ambiguous: identity vs flip"])
    if preset.startswith("Guided"):
        target_rule="Rotate 90°"; demos_n=3; noise=0.0
    elif preset.startswith("Hard"):
        target_rule="Rotate 90°"; demos_n=2; noise=0.10
    else:
        target_rule="Flip horizontal"; demos_n=1; noise=0.05
    target_rule = st.selectbox("Hidden rule in the toy world", [r.name for r in RULES], index=[r.name for r in RULES].index(target_rule))
    demos_n = st.slider("Demonstrations", 1, 5, demos_n)
    noise = st.slider("Demonstration noise", 0.0, 0.30, noise, 0.01)
    iterations = st.slider("Latent refinement steps", 1, 16, 6)
    strength = st.slider("Update strength α", 0.05, 1.00, 0.65, 0.05)
    temperature = st.slider("Readout temperature", 0.10, 2.00, 0.50, 0.05)
    seed = st.number_input("Seed", 0, 9999, 17)

rule,demos,(query,truth)=build_case(target_rule,demos_n,noise,int(seed))
result=run_recurrent_reasoner(demos,query,iterations,strength,temperature,noise*0.35)
pred=result["prediction"]
correct=np.array_equal(pred,truth)

st.markdown("## 1 · Watch latent reasoning happen")
col1,col2,col3=st.columns([1.4,1.4,1])
with col1:
    st.markdown("#### Demonstrations")
    for i,(x,y) in enumerate(demos,1):
        a,b=st.columns(2)
        with a: show_grid(x,f"Input {i}")
        with b: show_grid(y,f"Output {i}")
with col2:
    st.markdown("#### New query")
    a,b=st.columns(2)
    with a: show_grid(query,"Query input")
    with b: show_grid(truth,"Ground truth")
    st.divider()
    show_grid(pred,"Toy latent reasoner prediction")
    st.metric("Prediction", "✓ Correct" if correct else "✗ Wrong", delta=f"rule: {result['rule'].name}")
with col3:
    st.markdown("#### What is in the latent state?")
    probs=result["probabilities"]
    df=pd.DataFrame({"Rule":[r.name for r in RULES],"Belief":probs}).sort_values("Belief",ascending=True)
    fig=px.bar(df,x="Belief",y="Rule",orientation="h",range_x=[0,1],title="Final hypothesis belief")
    fig.update_layout(height=430,margin=dict(l=10,r=10,t=45,b=10))
    st.plotly_chart(fig,use_container_width=True)

st.markdown("### 2 · Scrub the recurrence")
traj=result["trajectory"]
traj_df=pd.DataFrame(traj,columns=[r.name for r in RULES])
traj_df["Step"]=np.arange(len(traj_df))
long=traj_df.melt(id_vars="Step",var_name="Hypothesis",value_name="Belief")
fig=px.line(long,x="Step",y="Belief",color="Hypothesis",markers=True,title="The state is a fixed vector; reasoning is repeated refinement")
fig.update_layout(height=460,legend_title_text="")
st.plotly_chart(fig,use_container_width=True)

peak_idx=int(np.argmax(result["probabilities"]))
second=int(np.argsort(result["probabilities"])[-2])
margin=float(result["probabilities"][peak_idx]-result["probabilities"][second])
c1,c2,c3,c4=st.columns(4)
c1.metric("Winning hypothesis",result["rule"].name)
c2.metric("Confidence",f"{probs[peak_idx]*100:.1f}%")
c3.metric("Margin vs #2",f"{margin*100:.1f} pp")
c4.metric("Visible reasoning trace","0 verbal steps")

st.markdown("### 3 · Challenge the claim")
st.info("Try the noisy preset, lower the number of demonstrations, and sweep the refinement steps. Notice that recurrence can sharpen a decision when evidence is consistent, but it cannot manufacture evidence that the demonstrations never contained.")

st.markdown("## 4 · BDH / BDH-CQ bridge")
left,right=st.columns([1.1,1])
with left:
    st.markdown("### From attention to synaptic memory")
    st.markdown("This panel is a **toy visualization of the published BDH construction**, not the official model. The published derivation rewrites attention so a fixed high-dimensional state can be read as a neuron/synapse graph; the synaptic state is updated by an outer-product write.")
    st.latex(r"\sigma_t = \sigma_{t-1} + x_t^\top v_t, \qquad o_t = x_t\sigma_t")
    st.markdown("In the BDH framing, the update is Hebbian-like: a pair of co-active patterns strengthens a connection. That makes memory part of the computational fabric rather than a separate growing token list.")
    st.markdown("**Important distinction:** this app does not reproduce BDH weights, training, or performance. It uses the equation as an educational bridge.")
with right:
    st.markdown("### Watch synapses accumulate")
    bdh_steps=st.slider("Memory writes",1,12,7,key="bdh")
    hist=toy_synapse_replay(seed=int(seed),neurons=12,steps=bdh_steps)
    mat=hist[-1]
    fig=px.imshow(mat,text_auto=False,aspect="equal",title=f"Toy synaptic state after {bdh_steps} writes",color_continuous_scale="Viridis")
    fig.update_layout(height=430,margin=dict(l=10,r=10,t=45,b=10),coloraxis_colorbar_title="strength")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Toy computation: sparse random x/v patterns produce outer-product updates. This visualizes the mechanism, not BDH behavior.")

st.markdown("## 5 · Why this is frontier work")
compare=pd.DataFrame([
    ["Transformer + CoT","Reason through language tokens","Growing textual trace","Observable, verbose"],
    ["Coconut (2024)","Continuous hidden-state reasoning","Latent states / reused hidden state","Less verbalization; latent search"],
    ["HRM (2025)","Recurrent multi-timescale reasoning","Recurrent hidden state","No explicit CoT"],
    ["BDH-CQ (2026)","Context updates memory + recurrent latent reasoning","Recurrent memory + latent iterations","No verbalized intermediate reasoning"],
])
compare.columns=["Family / system","Core idea","Where computation lives","Teaching takeaway"]
st.dataframe(compare,use_container_width=True,hide_index=True)

st.markdown("## 6 · 60-second challenge")
st.markdown("**Predict before reading the answer:** What happens when you increase recurrence from 1 → 16 while the demonstrations are noisy? Does the model become more certain, more accurate, both, or neither? Use the controls above, then explain the result in your own words.")

with st.expander("Evidence, scope & provenance"):
    st.markdown("""
**Selected primary sources (2024–2026):**

1. Hao et al., *Training Large Language Models to Reason in a Continuous Latent Space* (COLM 2025; arXiv:2412.06769). The paper introduces Coconut, feeding hidden states back as continuous thoughts rather than decoding them as words.
2. Kosowski et al., *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain* (arXiv:2509.26507, 2025). The paper introduces BDH and describes sparse positive activations, Hebbian working memory, and the GPU-friendly formulation.
3. Wang et al., *Hierarchical Reasoning Model* (arXiv:2506.21734, 2025). The paper studies recurrent reasoning with latent state and multi-timescale modules.
4. Engdahl et al., *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning* (arXiv:2608.09888, 2026). The paper describes inference-time recurrent memory updates from demonstrations and iterative latent reasoning without verbalizing intermediate reasoning.

**Evidence labels:** benchmark and paper claims belong to the cited sources; the toy solver and plots in this app are original educational computations; the BDH matrix panel is a mechanistic visualization of the published equations, not an official implementation.
""")

st.markdown("---")
st.caption("Latent Frontier Lab · educational artifact · built for the DataForge Pathway Track brief. Read the README before submission.")
