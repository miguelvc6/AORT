# AORT Implementation Plan

This document is written for two audiences at once:

- **Human researcher**: to decide what to build and in what order
- **Coding agent**: to implement the model with minimal ambiguity

The goal is to get to a valid first experimental result quickly, while keeping the scientific interpretation clean.

---

## 1. Objective

Implement **AORT-v0** on top of the TRM codebase as a minimal proof of concept.

### AORT-v0 definition

- recursive model
- operator routing between `MLP` and `Attention`
- fixed number of loops `K`
- no unified STOP operator yet
- use existing task loss
- add routing diagnostics

### AORT-v1 definition

- same as v0
- optionally re-enable the existing halting mechanism already present in the TRM codebase

### AORT-v2 definition

- unify operator routing and early exit into one controller

Only move to `v1` after `v0` works.
Only move to `v2` after `v1` is understood.

---

## 2. Use the TRM repository as a starting base

For the first prototype, this is worth it because the repo already contains:

- the recursive carry structure
- the training loop
- task loaders
- MLP-vs-attention block variants
- an existing halting mechanism

---

## 3. Research constraints

These are not optional.
They define the experiment.

### 3.1 Do not change too many things at once

The order must be:

1. reproduce TRM baselines
2. add operator routing with fixed `K`
3. compare against matched-parameter fixed baselines
4. add halting later

### 3.2 Fairness requirement

Because AORT instantiates more than one operator branch, it must be compared to baselines with the **same operator library instantiated**.

Required comparison set:

- `DualBranch-FixedMLP`
- `DualBranch-FixedAttn`
- `DualBranch-FixedSchedule`
- `AORT-v0`

If this is not done, any gain can be dismissed as extra capacity.

### 3.3 First task should be easy to iterate on

Start with **Sudoku**.
Do not start with ARC.
Use Maze only as a second task once the implementation is stable.

---

## 4. Deliverables

### 4.1 Deliverable A: reproducible baselines

A researcher or agent should be able to run:

- fixed MLP TRM baseline
- fixed attention TRM baseline

and obtain sensible training curves.

### 4.2 Deliverable B: AORT-v0 implementation

A researcher or agent should be able to run:

- dual-branch fixed MLP
- dual-branch fixed attention
- dual-branch fixed schedule
- learned routed model

using the same trainer.

### 4.3 Deliverable C: diagnostics

The implementation must log:

- router logits / probabilities
- operator usage counts
- operator usage by loop index
- average routing entropy
- average effective recursion depth

---

## 5. Suggested repository workflow

### 5.1 Branches

Use branches like:

- `baseline-repro`
- `aort-v0-routing`
- `aort-v1-halting`
- `analysis-routing`

### 5.2 Commit discipline

Recommended commit boundaries:

1. baseline reproduction only
2. refactor block interface only
3. add dual-branch block without learning
4. add fixed routing modes
5. add learned router
6. add diagnostics
7. add config flags
8. add analysis scripts

Do not combine architecture changes and evaluation scripts in the same commit.

---

## 6. File-level plan

This section is the most important for coding agents.

### 6.1 Files likely to change

Primary:

- `models/recursive_reasoning/trm.py`
- relevant config files under `config/`
- possibly evaluator / logging utilities if metrics need to be surfaced

Secondary:

- helper utilities only if required for logging or config parsing

Avoid editing the full training stack unless blocked.

### 6.2 New files to consider

Optional but recommended:

- `models/recursive_reasoning/aort_router.py`
- `models/recursive_reasoning/aort_utils.py`
- `analysis/analyze_aort_routing.py`

However, if the repo style prefers keeping model logic in `trm.py`, use minimal changes there first.

---

## 7. Concrete implementation phases

## Phase 0 — read and reproduce

### Human intent

Understand the current TRM codepath before editing.

### Agent tasks

1. locate the current recurrent block implementation
2. identify where `mlp_t` selects MLP vs attention path
3. locate the halting head and carry update
4. run one short baseline training job
5. confirm outputs, logs, and checkpoints work

### Acceptance criteria

- baseline runs without crashes
- one short training job produces metrics and checkpoints
- agent can describe where to insert routing logic

---

## Phase 1 — refactor block interface minimally

### Human intent

Separate “operator implementation” from “operator choice”.

### Agent tasks

Refactor the existing block into reusable subcomponents:

- `MLPOperator`
- `AttentionOperator`
- wrapper block that currently chooses one of them

The refactor must preserve behavior for existing configs.

### Acceptance criteria

- old configs still work unchanged
- `TRM-MLP` reproduces previous behavior
- `TRM-Attn` reproduces previous behavior

### Notes

This is the point where many implementations go wrong: they rewrite too much.
Do not rewrite the carry, trainer, or dataset logic here.

---

## Phase 2 — dual-branch fixed routing

### Human intent

Create fair baselines before learned routing.

### Agent tasks

Implement a dual-branch block that instantiates both operator branches and supports these routing modes:

- `fixed_mlp`
- `fixed_attn`
- `fixed_schedule`

### Required config flags

```yaml
operator_routing: true
routing_mode: fixed_mlp | fixed_attn | fixed_schedule | learned_soft | learned_st
schedule_type: mlp_then_attn
```

### Fixed schedule examples

- first half loops MLP, second half Attention
- alternating MLP / Attention

### Acceptance criteria

- each routing mode runs
- outputs have correct shape
- fixed routing is deterministic and testable

---

## Phase 3 — learned soft router

### Human intent

Test the core hypothesis with the most stable routing method.

### Agent tasks

Add a small router network:

```text
input: pooled z_H, pooled z_L, step embedding
output: 2 logits
```

Recommended default router:

- MLP with one hidden layer
- GELU activation
- no large router

### Suggested implementation details

- router input uses one summary per example
- use mean pooling or first control token
- softmax temperature configurable
- collect router logits and probabilities for logging

### Acceptance criteria

- learned routing runs end-to-end
- probabilities sum to one
- training remains numerically stable
- metrics and router diagnostics are logged

---

## Phase 4 — diagnostics and analysis hooks

### Human intent

Know whether the router learned anything meaningful.

### Agent tasks

Log at training and evaluation time:

- average probability of each operator
- entropy per loop
- histogram of selected operator per loop
- solved vs unsolved routing statistics if labels are available

Save a compact analysis artifact, e.g. JSON or CSV:

```json
{
  "step": ...,
  "loop_idx": ...,
  "p_mlp_mean": ...,
  "p_attn_mean": ...,
  "entropy_mean": ...
}
```

### Acceptance criteria

- post-hoc script can plot operator usage over loops
- researcher can detect collapse or trivial schedules

---

## Phase 5 — optional hard routing

### Human intent

Test whether discrete selection matters beyond soft mixing.

### Agent tasks

Add optional straight-through Gumbel-softmax:

- configurable by flag
- disabled by default
- same diagnostics as soft routing

### Acceptance criteria

- no NaN issues
- fallback to soft routing remains available

Hard routing is not required for the first research result.

---

## Phase 6 — optional existing halting integration

### Human intent

Check whether operator routing still helps when adaptive stopping is active.

### Agent tasks

Reuse the existing halting mechanism.
Do not merge STOP into the operator router yet.

### Acceptance criteria

- routed model works with previous halting path enabled
- halting statistics are logged separately from operator routing

---

## Phase 7 — unified STOP controller (v2)

### Human intent

Only attempt this after v0/v1 have produced interpretable results.

### Agent tasks

Extend router from 2 actions to 3:

- `MLP`
- `ATTN`
- `STOP`

or implement a separate controller if cleaner.

### Acceptance criteria

- training is stable
- stop frequency is reasonable
- not attempted before v0 comparison is complete

---

## 8. Pseudocode target for coding agents

```python
class DualBranchOperatorBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp_op = MLPOperator(config)
        self.attn_op = AttentionOperator(config)
        self.router = SmallRouter(config) if config.routing_mode.startswith("learned") else None

    def route_probs(self, state, loop_idx):
        if self.config.routing_mode == "fixed_mlp":
            return one_hot_probs(state, idx=0, num_ops=2)
        if self.config.routing_mode == "fixed_attn":
            return one_hot_probs(state, idx=1, num_ops=2)
        if self.config.routing_mode == "fixed_schedule":
            idx = schedule_to_idx(loop_idx, self.config.schedule_type)
            return one_hot_probs(state, idx=idx, num_ops=2)
        logits = self.router(summary_from_state(state, loop_idx))
        return soft_or_st_probs(logits, self.config)

    def forward(self, state, loop_idx):
        probs, aux = self.route_probs(state, loop_idx)
        out_mlp = self.mlp_op(state)
        out_attn = self.attn_op(state)
        next_state = mix_states(out_mlp, out_attn, probs)
        return next_state, aux
```

This target is conceptual.
Preserve the actual TRM state structure where possible.

---

## 9. Human runbook

### 9.1 Step-by-step order

1. run short `TRM-MLP` baseline
2. run short `TRM-Attn` baseline
3. confirm gap direction is sensible
4. implement dual-branch fixed routes
5. run `fixed_mlp`, `fixed_attn`, `fixed_schedule`
6. implement learned soft router
7. compare all four models
8. inspect routing logs before touching halting

### 9.2 What to look for

Good signs:

- learned routed model beats best fixed dual-branch baseline
- operator usage differs across loop index or examples
- entropy declines gradually, not instantly
- fixed schedule does not fully match learned routing

Bad signs:

- immediate collapse to one operator
- learned router matches fixed schedule exactly
- dual-branch fixed baseline erases the gain
- training instability appears only when routing is enabled

---

## 10. Metrics and experiment table

Create a simple results table with these columns:

| model              | params | FLOPs proxy | task metric | avg depth | router entropy | p(MLP) | p(Attn) |
| ------------------ | -----: | ----------: | ----------: | --------: | -------------: | -----: | ------: |
| TRM-MLP            |        |             |             |           |                |        |         |
| TRM-Attn           |        |             |             |           |                |        |         |
| Dual-FixedMLP      |        |             |             |           |                |        |         |
| Dual-FixedAttn     |        |             |             |           |                |        |         |
| Dual-FixedSchedule |        |             |             |           |                |        |         |
| AORT-v0            |        |             |             |           |                |        |         |

For `TRM-MLP` and `TRM-Attn`, routing columns can be blank or trivial.

---

## 11. Minimal experiment plan

### Stage A — smoke tests

- 1 short run per baseline
- small training budget
- verify correctness, not final performance

### Stage B — core comparison

- full-ish run on Sudoku for all fixed and learned routing modes
- same seed count where feasible
- compare matched-parameter variants first

### Stage C — second regime

Choose one:

- Maze short run
- another Sudoku regime
- halting-enabled AORT-v1

Only do Stage C if Stage B is interpretable.

---

## 12. Risk register

### Risk 1: implementation overreach

Symptom:

- too many files changed
- baseline breaks

Response:

- return to smallest working patch
- preserve old codepath exactly

### Risk 2: unfair comparison

Symptom:

- learned routing only compared to single-branch baselines

Response:

- add fixed dual-branch baselines immediately

### Risk 3: routing collapse

Symptom:

- p(MLP) ≈ 1 or p(Attn) ≈ 1 very early

Response:

- add entropy regularization
- use higher temperature
- delay hard routing

### Risk 4: interpretation failure

Symptom:

- gain exists, but fixed schedule matches it

Response:

- narrow the claim: stage-specific operators, not input-adaptive routing

---

## 13. Acceptance criteria for the first milestone

The first milestone is complete if all are true:

1. baseline TRM runs reproduced in the chosen repo branch
2. dual-branch fixed baselines run correctly
3. learned soft-routed AORT-v0 trains without instability
4. routing diagnostics are logged and analyzable
5. at least one results table compares fixed vs learned routing fairly

Only after that should additional complexity be added.

---

## 14. Suggested prompts for coding agents

### Prompt A — baseline inspection

```text
Inspect the TRM implementation and identify the smallest set of files needed to add operator routing between the existing MLP-style and attention-style recurrent block variants. Do not change the training loop yet. Summarize the current block interface, carry structure, and halting path.
```

### Prompt B — safe refactor

```text
Refactor the current TRM block implementation so that the MLP-style operator and attention-style operator are separate reusable modules, while preserving exact behavior for the existing configs. Minimize the diff and do not change trainer behavior.
```

### Prompt C — dual-branch fixed routing

```text
Implement a dual-branch recurrent block that instantiates both the MLP-style and attention-style operators and supports routing modes fixed_mlp, fixed_attn, and fixed_schedule. Add config flags, keep shapes unchanged, and preserve backward compatibility.
```

### Prompt D — learned soft routing

```text
Add a small learned router that takes a pooled state summary and loop index and outputs 2 logits for MLP vs attention. Use soft routing by default. Log router probabilities and entropy. Keep halting unchanged.
```

### Prompt E — diagnostics

```text
Add evaluation-time logging and artifact saving for operator usage frequency, usage by loop index, and average routing entropy. Produce a compact CSV or JSON that can be plotted later.
```

---

## 15. Final recommendation

The correct implementation philosophy is:

- **minimal patch**
- **fair baselines first**
- **learned routing before learned stopping**
- **analysis hooks from day one**

That is the shortest path to a result that is both technically valid and scientifically interpretable.
