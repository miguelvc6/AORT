# AORT Human Runbook Commands

These commands follow the order in `agents/aort_implementation_plan.md`: baseline first, matched dual-branch fixed routes second, learned AORT-v0 third, then routing-log inspection.

Assumptions:

- Run from the repository root.
- Use the existing Sudoku dataset at `data/sudoku-extreme-full`.
- Use one GPU.
- Keep `act_enabled: False` for the v0 comparison configs.
- Use the same seed and training budget for every model in a comparison group.

## 1. Smoke Runs

Use these first to verify every model path trains, evaluates, checkpoints, and writes usable routing diagnostics where applicable.

```bash
uv run python pretrain.py \
  arch=trm_mlp_fixedk \
  data_paths="[data/sudoku-extreme-full]" \
  evaluators="[]" \
  epochs=1 eval_interval=1 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  global_batch_size=256 \
  +run_name=smoke_trm_mlp_fixedk

uv run python pretrain.py \
  arch=trm_attn_fixedk \
  data_paths="[data/sudoku-extreme-full]" \
  evaluators="[]" \
  epochs=1 eval_interval=1 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  global_batch_size=256 \
  +run_name=smoke_trm_attn_fixedk

uv run python pretrain.py \
  arch=aort_fixed_mlp \
  data_paths="[data/sudoku-extreme-full]" \
  evaluators="[]" \
  epochs=1 eval_interval=1 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  global_batch_size=256 \
  +run_name=smoke_dual_fixed_mlp

uv run python pretrain.py \
  arch=aort_fixed_attn \
  data_paths="[data/sudoku-extreme-full]" \
  evaluators="[]" \
  epochs=1 eval_interval=1 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  global_batch_size=256 \
  +run_name=smoke_dual_fixed_attn

uv run python pretrain.py \
  arch=aort_fixed_schedule \
  data_paths="[data/sudoku-extreme-full]" \
  evaluators="[]" \
  epochs=1 eval_interval=1 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  global_batch_size=256 \
  +run_name=smoke_dual_fixed_schedule

uv run python pretrain.py \
  arch=aort_v0 \
  data_paths="[data/sudoku-extreme-full]" \
  evaluators="[]" \
  epochs=1 eval_interval=1 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  global_batch_size=256 \
  +run_name=smoke_aort_v0_learned_soft
```

## 2. Inspect Routing Stats

After the routed smoke runs, confirm that loop stats are non-empty.

```bash
uv run python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("checkpoints").glob("**/routing_stats_step_*.json")):
    data = json.loads(path.read_text())
    print(path)
    for set_name, stats in data.get("sets", {}).items():
        print(" ", set_name, stats.get("summary"), "loops:", len(stats.get("loops", [])))
PY
```

Expected smoke-run result:

- `aort_fixed_mlp`: `router_p_mlp` near `1.0`, `router_p_attn` near `0.0`.
- `aort_fixed_attn`: `router_p_mlp` near `0.0`, `router_p_attn` near `1.0`.
- `aort_fixed_schedule`: non-empty per-loop rows with early loops using MLP and later loops using attention.
- `aort_v0`: non-empty per-loop rows and nonzero entropy early in training.

## 3. Inspect Metric History

Every new run appends train, eval, and initialization metrics to:

```text
checkpoints/<project_name>/<run_name>/metrics.jsonl
```

The file has one JSON object per logged step. Train keys are flat, e.g. `train/lm_loss`, and eval keys are flattened under `eval/`, e.g. `eval/all/exact_accuracy`, `eval/all/router_entropy`, and `eval/ARC/pass@1` when ARC evaluators are enabled.

## 4. Core Comparison Runs

Use the same budget for all six models. This mirrors the README Sudoku settings while keeping the comparison matched.

```bash
uv run python pretrain.py \
  arch=trm_mlp_fixedk \
  data_paths="[data/sudoku-extreme-full]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  global_batch_size=256 ema=True seed=0 \
  +run_name=core_trm_mlp_fixedk_seed0

uv run python pretrain.py \
  arch=trm_attn_fixedk \
  data_paths="[data/sudoku-extreme-full]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  global_batch_size=256 ema=True seed=0 \
  +run_name=core_trm_attn_fixedk_seed0

uv run python pretrain.py \
  arch=aort_fixed_mlp \
  data_paths="[data/sudoku-extreme-full]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  global_batch_size=256 ema=True seed=0 \
  +run_name=core_dual_fixed_mlp_seed0

uv run python pretrain.py \
  arch=aort_fixed_attn \
  data_paths="[data/sudoku-extreme-full]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  global_batch_size=256 ema=True seed=0 \
  +run_name=core_dual_fixed_attn_seed0

uv run python pretrain.py \
  arch=aort_fixed_schedule \
  data_paths="[data/sudoku-extreme-full]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  global_batch_size=256 ema=True seed=0 \
  +run_name=core_dual_fixed_schedule_seed0

uv run python pretrain.py \
  arch=aort_v0 \
  data_paths="[data/sudoku-extreme-full]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  global_batch_size=256 ema=True seed=0 \
  +run_name=core_aort_v0_learned_soft_seed0
```

## 5. Repeat Seeds

If the seed-0 result is interpretable, repeat the core comparison with the same commands and change only:

```bash
seed=1
+run_name=..._seed1
```

then:

```bash
seed=2
+run_name=..._seed2
```

## 6. Do Not Start Halting Yet

Only run AORT-v1 or unified STOP experiments after the table contains:

- `TRM-MLP`
- `TRM-Attn`
- `Dual-FixedMLP`
- `Dual-FixedAttn`
- `Dual-FixedSchedule`
- `AORT-v0`

and the routed runs have non-empty routing stats.
