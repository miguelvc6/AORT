# Adaptive Operator Recursive Transformer (AORT)

## 1. Purpose

AORT is a recursive reasoning model inspired by Tiny Recursive Models (TRM), adaptive computation, and conditional computation. The core idea is to **select the update operator at each recursive step** instead of repeating a fixed operator for all loops.

In the minimal version, the model chooses between:

- an **MLP-style recurrent update**
- an **attention-based recurrent update**
- optionally, a **STOP / early-exit action**

The intended research question is:

> Does adaptive operator selection inside recursive computation improve generalization and/or compute efficiency over a fixed recurrent operator?

This specification is written for a first research prototype, not for a production-scale language model.

---

## 2. Scope

### 2.1 In scope

- Recursive reasoning over a bounded number of loops `K`
- Operator routing at each loop
- Optional early exit
- Compatibility with TRM-style hidden-state recurrence
- Initial experiments on algorithmic / reasoning tasks such as Sudoku or Maze

### 2.2 Out of scope for v1

- Token-level MoE routing
- Distributed expert parallelism
- Large-scale LLM pretraining
- Many-expert libraries
- Learned memory beyond the recurrent hidden state
- Jointly changing task format, dataset, and training objective at the same time

---

## 3. High-level model summary

AORT maintains a recurrent hidden state and repeatedly refines it for up to `K` loops.
At loop `k`, a router examines the current state and selects one operator from a small library.
The selected operator transforms the state.
The process either continues to the next loop or stops early.

### 3.1 Intuition

Different recursion steps may require different inductive biases:

- **MLP-style mixing** may be enough for short-context local refinement
- **Attention** may be useful when the state must integrate information across positions
- **Early exit** may be correct once the current state is already sufficient

This makes AORT an **adaptive operator recursion** model rather than a standard mixture-of-experts model.

---

## 4. Naming

### 4.1 Family name

**Adaptive Operator Recursive Transformer (AORT)**

### 4.2 Recommended experiment labels

- `TRM-MLP`: fixed MLP-style TRM baseline
- `TRM-Attn`: fixed attention-style TRM baseline
- `AORT-v0`: routed operator choice, no early exit integration
- `AORT-v1`: routed operator choice + existing halting head
- `AORT-v2`: unified operator + STOP controller

---

## 5. Architectural definition

### 5.1 Inputs

A batch item contains at minimum:

- `x`: tokenized input sequence
- optional task / puzzle identifier embedding
- optional metadata needed by the task loader

The exact task interface is inherited from the base TRM setup.

### 5.2 Outputs

The model outputs:

- token logits or task logits
- optional halting logits / stop decision statistics
- optional router logits and routing diagnostics

### 5.3 State

The model maintains a recurrent carry:

- `z_H`: high-level latent state
- `z_L`: low-level latent state
- loop counter `k`
- halting status

For direct TRM compatibility, the first prototype should preserve the existing carry structure.

---

## 6. Operator library

### 6.1 Minimal library for v0/v1

Use exactly two operators:

1. **MLP operator**
   - same input/output shape as the recurrent state
   - intended to match the current TRM MLP-style update path

2. **Attention operator**
   - same input/output shape as the recurrent state
   - intended to match the current TRM attention-style update path

### 6.2 Optional v2 extension

3. **STOP operator**
   - indicates no further recursive update is needed
   - may either trigger immediate exit or map to a no-op plus halt

### 6.3 Interface requirements

Every operator must satisfy:

```text
f_i: R^(B x L x D) -> R^(B x L x D)
```

All operators must:

- preserve batch, sequence, and hidden dimensions
- be residual-compatible
- operate on the same recurrent state representation
- support the same dtype / device flow

---

## 7. Router

### 7.1 Router purpose

The router selects which operator to apply at loop `k`.

It should answer:

> Given the current state, which update rule is most appropriate now?

### 7.2 Router input

For v0/v1, the router input should be simple and stable:

- pooled summary of `z_H`
- optionally pooled summary of `z_L`
- loop index embedding or scalar step feature
- optional task identifier embedding

Recommended default:

```text
r_k_input = concat(pool(z_H), pool(z_L), step_embedding(k))
```

where `pool` is one of:

- first-token summary
- mean pooling
- designated control position

Do not start with token-level routing.

### 7.3 Router output

#### v0 / v1

Router outputs two logits:

```text
g_k = Router(r_k_input) ∈ R^2
```

corresponding to:

- `MLP`
- `Attention`

#### v2

Router outputs three logits:

```text
g_k = Router(r_k_input) ∈ R^3
```

corresponding to:

- `MLP`
- `Attention`
- `STOP`

### 7.4 Routing modes

#### Soft routing (recommended first)

```text
p_k = softmax(g_k / tau)
h_{k+1} = p_k[MLP] * f_MLP(h_k) + p_k[ATTN] * f_ATTN(h_k)
```

Pros:

- stable optimization
- easy diagnostics
- simpler training than discrete routing

Cons:

- not truly sparse
- less faithful to discrete algorithm selection

#### Hard routing with straight-through Gumbel-softmax

Recommended only after the soft version works.

### 7.5 Routing granularity

For v0/v1, routing is:

- **per-example**
- **per-loop**
- **global across positions**

This keeps the experiment aligned with the core hypothesis.

---

## 8. Halting / early exit

### 8.1 v0

No learned early exit.
Run for exactly `K` loops.

Purpose:

- isolate operator routing from halting
- simplify interpretation of results

### 8.2 v1

Use the **existing halting head** from the TRM implementation.
This tests whether routing helps even when the halting mechanism is unchanged.

### 8.3 v2

Unify operator selection and stopping into one controller.

#### Option A: three-way router

```text
a_k ∈ {MLP, ATTN, STOP}
```

#### Option B: decoupled controllers

- router selects operator
- separate halt head decides continue/stop

Recommended order:

1. v0
2. v1
3. v2

---

## 9. Forward pass specification

Let `h_k` denote the recurrent hidden state representation used by the selected operator.
For TRM-compatible implementation, `h_k` may correspond to a structured state `(z_H, z_L)` rather than one tensor.

### 9.1 v0 forward pass

```text
h_0 = Encode(x)
for k in 0..K-1:
    r_k = Router(summary(h_k), k)
    p_k = softmax(r_k / tau)
    h_mlp = f_MLP(h_k)
    h_attn = f_ATTN(h_k)
    h_{k+1} = p_k[0] * h_mlp + p_k[1] * h_attn

y = Decode(h_K)
```

### 9.2 v1 forward pass

Same as v0, but after each loop the existing halting module may terminate the trajectory.

### 9.3 v2 forward pass

```text
h_0 = Encode(x)
for k in 0..K-1:
    a_k = Controller(summary(h_k), k)
    if a_k == STOP:
        break
    elif a_k == MLP:
        h_{k+1} = f_MLP(h_k)
    elif a_k == ATTN:
        h_{k+1} = f_ATTN(h_k)

y = Decode(h_final)
```

---

## 10. Training objective

### 10.1 Main task loss

Use the same supervised objective as the base task.
Examples:

- token-level cross-entropy
- exact output loss
- task-specific structured loss

Denote it by:

```text
L_task
```

### 10.2 Router regularization

#### Entropy regularization

Purpose:

- avoid immediate routing collapse
- encourage exploration early in training

```text
L_entropy = - mean_k H(p_k)
```

Use with a small weight and possibly anneal down.

#### Load-balance regularization

Optional.
Useful if routing collapses strongly to one operator.

```text
L_balance = sum_i (mean_batch(p_k[i]) - target_i)^2
```

with `target_i = 1 / num_operators` unless a prior is desired.

### 10.3 Compute / halting regularization

For early-exit variants:

```text
L_compute = mean(number_of_steps_used)
```

or a normalized compute penalty.

### 10.4 Full objective

For v0:

```text
L = L_task + λ_ent * L_entropy + λ_bal * L_balance
```

For v1/v2:

```text
L = L_task + λ_ent * L_entropy + λ_bal * L_balance + λ_comp * L_compute
```

---

## 11. Fair comparison constraints

The routed model must be compared fairly.
The biggest confounder is extra parameter count.

### 11.1 Required matched-parameter baselines

When the routed model contains both operators, compare against:

1. **Dual-branch fixed MLP**
   - instantiate both branches
   - always choose MLP

2. **Dual-branch fixed Attention**
   - instantiate both branches
   - always choose Attention

3. **Dual-branch fixed schedule**
   - e.g. MLP for early loops, Attention for later loops

4. **Learned routed AORT**

These are better baselines than comparing only to single-branch TRM.

### 11.2 Keep one variable changing at a time

Order of experimentation:

1. fixed operator TRM baselines
2. routed operator, fixed `K`
3. routed operator + existing halting
4. routed operator + unified STOP controller

---

## 12. Diagnostics and logging

At minimum, log:

- train loss
- validation loss
- task accuracy / exact match
- average recursion depth
- router entropy
- operator usage frequency
- operator usage by loop index
- operator usage for solved vs unsolved examples
- halting distribution (if enabled)

Useful derived analyses:

- does routing depend on step or on input?
- does the model collapse to one operator?
- is the learned router better than a fixed schedule?

---

## 13. Evaluation protocol

### 13.1 First task

Start with a task where TRM-like differences are already known to matter.
Sudoku-style short fixed-context reasoning is a good first setting.

### 13.2 Second task

Use a larger-context or relational task where attention may be more helpful.
Maze-style reasoning is a good second candidate.

### 13.3 Minimum success criteria

AORT is promising only if all of the following are true:

1. it beats the best matched-parameter fixed baseline
2. the router does not collapse trivially
3. the gain is not reproducible by a fixed schedule
4. the result transfers to at least one second regime or task

---

## 14. Recommended ablations

### 14.1 Core ablations

- `TRM-MLP`
- `TRM-Attn`
- `DualBranch-FixedMLP`
- `DualBranch-FixedAttn`
- `DualBranch-FixedSchedule`
- `AORT-v0`

### 14.2 Router ablations

- loop-index-only router
- state-only router
- state + loop router
- soft vs hard routing

### 14.3 Halting ablations

- fixed `K`
- existing TRM halting
- unified STOP controller

### 14.4 Capacity ablations

- small hidden size
- standard hidden size
- small vs larger operator library

Do not expand the operator library before the 2-operator version is understood.

---

## 15. Failure modes

### 15.1 Router collapse

Symptoms:

- one operator selected almost always
- entropy quickly goes to zero

Mitigation:

- entropy regularization
- temperature schedule
- balanced warm-up

### 15.2 Static schedule masquerading as routing

Symptoms:

- per-step usage is nearly deterministic
- little dependence on input state

Mitigation:

- compare against fixed schedules
- analyze routing conditional on input categories

### 15.3 Capacity-only gains

Symptoms:

- learned routing only beats single-branch baselines
- fixed dual-branch baselines perform similarly

Mitigation:

- matched-parameter baselines
- FLOP-aware comparisons

### 15.4 Optimization instability

Symptoms:

- noisy training
- route oscillation
- poor convergence

Mitigation:

- start with soft routing
- keep halting fixed in v0
- reduce simultaneous changes

---

## 16. Minimal pseudocode

```python
class AORTRouter(nn.Module):
    def __init__(self, hidden_size, num_ops=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2 + hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_ops),
        )

    def forward(self, z_h_summary, z_l_summary, step_emb):
        x = torch.cat([z_h_summary, z_l_summary, step_emb], dim=-1)
        return self.net(x)


class AORTBlock(nn.Module):
    def __init__(self, mlp_block, attn_block, router):
        super().__init__()
        self.mlp_block = mlp_block
        self.attn_block = attn_block
        self.router = router

    def forward(self, state, step_emb, tau=1.0):
        z_h, z_l = state
        logits = self.router(pool(z_h), pool(z_l), step_emb)
        probs = torch.softmax(logits / tau, dim=-1)

        out_mlp = self.mlp_block(state)
        out_attn = self.attn_block(state)

        next_state = blend_states(out_mlp, out_attn, probs)
        return next_state, {"router_logits": logits, "router_probs": probs}
```

This pseudocode is conceptual.
The first implementation should reuse existing TRM internals instead of rebuilding the full architecture immediately.

---

## 17. Configuration schema

Suggested config additions:

```yaml
arch:
  name: aort
  hidden_size: ...
  num_heads: ...
  expansion: ...
  H_cycles: ...
  L_cycles: ...
  halt_max_steps: ...

  operator_routing: true
  operator_library: [mlp, attention]
  routing_mode: soft          # soft | st-gumbel | hard
  router_input: state+step    # step | state | state+step
  router_pooling: first_token # first_token | mean
  router_temp: 1.0
  router_entropy_coef: 0.001
  router_balance_coef: 0.0

  use_existing_halt_head: false
  unified_stop_controller: false
```

For v1:

```yaml
  use_existing_halt_head: true
```

For v2:

```yaml
  unified_stop_controller: true
```

---

## 18. Decision log

### v0 decision

- Use only 2 operators
- Keep fixed `K`
- Soft routing
- Reuse TRM carry and trainer

### v1 decision

- Re-enable existing halting mechanism
- Keep operator routing separate from halting controller

### v2 decision

- Try unified `{MLP, Attention, STOP}` controller

This staged plan is part of the model design, not a mere implementation convenience.
It is necessary for clean scientific interpretation.

---

## 19. Summary

AORT is a recursive reasoning model that chooses **which operator to apply at each loop** instead of repeating one fixed operator throughout recursion.

The first convincing result is not simply that it performs well.
The first convincing result is that:

- it beats the best fixed matched-parameter baseline,
- the routing is meaningfully nontrivial,
- and the gain cannot be reduced to extra capacity or a fixed hand-crafted schedule.

That is the bar this specification is designed to test.
