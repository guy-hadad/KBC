# Paper experiment campaign status

Last operational update: 2026-09-17 (Asia/Jerusalem).

## Scope and safety limit

- Declared experiment cells: **16,441**.
- CPU cells: **13,300** across 15 suite components.
- GPU cells: **3,141** across 9 suite components.
- User limit: at most **100 campaign jobs** in SLURM at once.
- Current design: four array elements per suite component, or at most 96
  experiment workers plus preparation/smoke prerequisites.
- `scripts/submit_paper_experiments.py` enforces the cap before submission.
- Every suite has an isolated result directory under
  `$KBC_OUTPUT_ROOT/paper_runs/<suite>/`.

The original 32-element arrays (`19952502`--`19952525`) were cancelled before
running because they exceeded the requested job cap. Most replacement CPU
arrays used IDs `19952542`--`19952566` (with gaps; consult the submission
manifest). The three CPU arrays still running near their 24-hour limit were
replaced by resumable arrays after preserving their completed results. The
first replacement GPU arrays were likewise replaced because their 24-hour
limit was too short for the GEM cells.

## Prerequisite gates

| Purpose | Job ID | State |
|---|---:|---|
| Prepare all paper datasets | `19952364` | completed, exit `0:0` |
| Representative CPU smoke | `19952496` | completed, exit `0:0` |
| SmolLM2 135M/360M GPU smoke | `19952497` | completed, exit `0:0` |

## Active resumable GPU arrays

| Suite | Job ID | Cells |
|---|---:|---:|
| Core reproduction | `19972174` | 432 |
| GEM context budget | `19972176` | 480 |
| GEM event text | `19972179` | 480 |
| GEM full banking | `19972181` | 420 |
| GEM main banking | `19972182` | 260 |
| GEM model scale | `19972183` | 480 |
| GEM reference datasets | `19972184` | 325 |
| GEM template/time/type | `19972185` | 120 |
| Temporal models | `19972187` | 144 |

Each array has four workers, a seven-day allocation, and a pre-timeout requeue
hook. A restarted worker reruns the same deterministic slice and skips result
JSON files already present. This keeps the worker count fixed while allowing
multi-day completion.

## Banking GEM priority run

Job `19972485` is a targeted `gem-main-banking` anchor containing four exact
paper-grid cells: `gem-time-byte-f32`, `n=2048`, seed 13 on BankSim, PaySim,
IBM AML, and MBD-mini. Its suite indices are 5, 70, 135, and 200. The job uses
the same five epochs, batch size 4, gradient accumulation 4, learning rate
`1e-3`, context length 64, bf16 setting, and result namespace as the complete
main GEM suite.

The other KBC GPU arrays are temporarily held so this anchor is the first
eligible KBC GPU allocation. On successful completion it releases all five
banking-only GEM arrays (`gem-context-budget`, `gem-event-text`,
`gem-full-banking`, `gem-main-banking`, and `gem-model-scale`). Job `19972488`
then releases the deferred general/reference GPU arrays after those five
banking campaigns complete. Existing unrelated account jobs are not modified.

## Resumed CPU arrays

| Suite | Job ID | Cells (including already-completed files) |
|---|---:|---:|
| Adaptation regimes | `19972237` | 1,080 |
| Core reproduction, CPU methods | `19972238` | 1,440 |
| Cross-schema transfer | `19972246` | 2,880 |

These arrays also have four seven-day, resumable workers. At replacement time,
most of their cells were already complete; workers skip those files and process
only the missing deterministic slice.

## Latest verified progress

At the 2026-08-09 audit, **12,878 result files** had `status: ok` and **zero
result files** had `status: failed`. Six CPU workers were still completing the
adaptation, core-reproduction, and cross-schema slices. GPU arrays were queued
or running subject to the account GPU QoS.

This is execution status, not a scientific conclusion. Results must still pass
completeness, metric-range, seed-aggregation, significance, and leakage audits
before entering a paper table. Paper-named methods marked as approximations in
the implementation manifest must continue to be reported as approximations.

All audited result files share source fingerprint
`9d5800dc3ddb8ac11ef87696264fd9379789cfe7bae8edbf2e076126bf778cb0` at
commit `58fe4dd2d9c789b35b2c4b1a432011111bc27b7e`. Python source is intentionally
frozen while the campaign runs; orchestration and documentation changes do not
enter that fingerprint.

The reporting pipeline was smoke-tested on all 580 completed `paper-headline`
cells and produced tables and CSV successfully. Its scaling-exponent page is
empty for that suite because `paper-headline` contains only the largest sample
size, while the generic prose assumes six scaling points. Do not publish that
page as-is; generate scaling fits from a suite with the full sample-size grid
and make the report prose suite-aware after the source-frozen run completes.

## Large-dataset scaling campaign (2026-09-07)

A second, independent campaign covering the three scale benchmarks. It shares
no result directory with the campaign above, so the audit trail of the 12 878
completed cells is untouched.

### Datasets prepared

| Dataset | Train | Val | Test | Marks | Mean events / seq | Natural pos. | Train pos. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthea EHR | 93 305 | 68 117 | 91 356 | 65 | 62.6 | 6.31 % | 20 % |
| Amazon Beauty 2014 | 34 023 | 7 891 | 10 460 | 10 | 6.0 | 26.31 % | 26.44 % |
| MBD (full) | 30 665 | 40 000 | 40 000 | 53 | 103.6 | 1.16 % | 20 % |

Amazon Beauty's training split is **not** rebalanced: its natural prevalence
already exceeds the 20 % target. Full MBD gives **10.2x** MBD-mini's training
pool (30 665 against 3 000), which is the point of the exercise: positives, not
clients, were the scarce resource.

Resolved scaling grids, ending at each whole pool:

| Dataset | Grid | Points |
| --- | --- | ---: |
| MBD (full) | 64 … 16 384, 30 665 | 10 |
| Synthea EHR | 64 … 65 536, 93 305 | 12 |
| Amazon Beauty 2014 | 64 … 16 384, 34 023 | 10 |

### Fixes this campaign required

These were latent breakages, not new work:

* **Every `slurm/*.sbatch` pointed at `REPO=/home/guyhada/KBC`, which does not
  exist.** The checkout is `/groups/bshapira_group/guyhada/KBC`. All scripts now
  use `${KBC_REPO:-/groups/bshapira_group/guyhada/KBC}`; the two `smoke_paper_*`
  scripts had the same path hard-coded in a `cd`.
* **`prepare_paper_data.sbatch` requested 96 GB**, above the 48 GB budget. Now
  48 GB, which the conversion fits after the memory work in
  `benchmark_design.md`.
* **polars needs a modern CPU node** — see `experiment_execution.md`. Job
  `21067512` died with `SIGILL` on `ise-cpu-intl-15` eight seconds in.
* **MBD-mini and full MBD name their targets differently** (`target_1..4`
  against `bcard/cred/zp/acquiring_target`). Resolved positionally, verified by
  matching per-client prevalence to within 0.02 pp.
* **`_wanted_members` matched nothing** on the first extraction attempt: archive
  members are `ptls/trx/...`, not `trx/...`, so a filtered `extractall` wrote an
  empty tree and still marked the step done. Extraction is now single-pass
  streaming and raises if a prefix filter matches no members.

### Prerequisite gates and arrays

| Purpose | Job ID | State at submission |
|---|---:|---|
| Convert the three scale datasets | `21067568` | completed, exit `0:0`, 7 min 36 s |
| Representative CPU smoke (10 cells) | `21067676` | completed, exit `0:0`, all cells `ok` |
| SmolLM2 GPU smoke | `21067677` | pending on `QOSMaxGRESPerUser` |

| Suite | Kind | Job ID | Cells | Workers |
|---|---|---:|---:|---:|
| `scale-datasets` | CPU (11 methods) | `21068281` | 1 920 | 16 |
| `scale-datasets` | GPU (3 LM entries) | `21068282` | 576 | 16 |
| `scale-datasets-baselines` | CPU (5 methods) | `21068283` | 864 | 16 |

**3 360 cells total, 2 784 CPU and 576 GPU.** Manifest:
`paper_runs/submissions/campaign_20260907T055701Z.json`.

The CPU arrays started immediately. The GPU array is deliberately deferred
behind the GPU smoke, which is itself queued behind the account's GPU QoS — the
previous campaign completed 12 878 CPU cells against roughly 20 of 3 141 GPU
cells, so the scaling curves are designed to be complete and publishable from
the CPU methods alone, with the three LM entries arriving whenever GPU frees up.

The CPU smoke ran on `ise-cpu-intl-15`, the same node whose missing AVX2 killed
the conversion. That is the intended asymmetry and confirms it: **running a cell
imports no polars**, so only prepare jobs need the node constraint.

### First-pass audit and the `Errno 116` race

At the first audit, **865 of 868 completed cells were `ok` and 3 had failed**,
all three with the same error:

```
OSError: [Errno 116] Stale file handle
  ... build_vocabulary -> load_jsonl_sequences(train_path)
```

All three were MBD at `n=30665`, its largest point. Cause: `subsampled_train_path`
published the subset cache with `os.replace`, which is atomic but **swaps the
inode**. With 32 workers converging on the same largest-`n` subset, one worker's
publish swapped the file while another already held the path open, and on NFS
that reader gets `ESTALE`.

Three changes, none of which alters any computed value:

1. `_publish_once` links the temp file into place with `os.link`, which fails if
   the target exists rather than replacing it. The file is created exactly once
   and never swapped under a reader. Subset contents are a pure function of
   (sample size, seed, source fingerprint), so first-writer-wins is safe.
2. `scripts/materialize_subsets.py` pre-builds every subset a suite needs,
   reading each training split once. Run before an array, workers then only
   *read* the cache, which removes the race entirely and also removes the
   per-cell cost of parsing a large split. All 96 subsets for this campaign are
   materialised.
3. `run_experiments.py --only-missing` (via `KBC_ONLY_MISSING=1` in the array
   scripts) runs only cells with no result or a result that is not `ok`, and
   overwrites, since `run_cell` otherwise returns any existing file untouched —
   which is why a failed cell is sticky and does not retry on its own.

Catch-up arrays `21092481` (scale-datasets) and `21092482`
(scale-datasets-baselines), four workers each, are queued `afterany` the main
arrays to fill whatever gaps they leave. The three failed result files were
deleted so they are rerun.

**Provenance note.** These fixes changed the source tree mid-campaign, so cells
carry two different `source_tree_sha256` values. The change is confined to
subset-cache concurrency and gap selection; no method, dataset, metric or seed
path is touched, so cells on either side of it are comparable. Do not treat the
fingerprint split as a data split.

### Progress at 2026-09-14

**2 934 of 3 360 cells complete, all `ok`, zero failures.** No `Errno 116` has
recurred since the `os.link` fix and the pre-materialised subset cache.

| Component | Job | Done | Remaining |
|---|---:|---:|---:|
| `scale-datasets-baselines` CPU | `21068283` | **864 / 864** | 0 |
| `scale-datasets` CPU | `21068281` | 1 690 / 1 920 | 230 |
| `scale-datasets` GPU | `21068282` | 380 / 576 | 196 |

The baselines gap-fill array `21092482` ran and found nothing to fill, which
confirms that suite is complete rather than silently short. `21092481` is still
queued behind the main CPU array.

**The GPU lane unblocked.** 380 of 576 LM cells are done against roughly 20 of
3 141 in the previous campaign, so the three LM entries will have full scaling
curves rather than the anchor points the deferred design assumed. The deferral
cost nothing and can be relaxed for future scale suites.

Interim tables, CSV and 496 figures are published to
[`../benchmark_scale/`](../benchmark_scale/), kept separate from
`../benchmark/` so the legacy 624-cell screen is not overwritten. Every number
there is provisional until the remaining 426 cells land.

One reporting fix went with it: `write_scaling_exponent_table` hard-coded
"fitted on six points per curve" and named PaySim in its prose, neither of which
is true for this suite. It now states the real per-dataset point counts, which is
what `docs/research/campaign_status.md` previously warned must be fixed before
that page is published.

### Progress at 2026-09-17

**3 013 of 3 360 cells complete, all `ok`, zero failures** (79 more than at
2026-09-14). Reports in [`../benchmark_scale/`](../benchmark_scale/) were
regenerated from these cells.

| Component | Job | Done | Remaining |
|---|---:|---:|---:|
| `scale-datasets-baselines` CPU | `21068283` | **864 / 864** | 0 |
| `scale-datasets` CPU | `21068281` | 1 768 / 1 920 | 152 |
| `scale-datasets` GPU | `21068282` (cancelled) | 381 / 576 | 195 |

Twelve of the 16 elements of `21068281` have exited `0:0`. Elements `_1` and
`_15` finished after the last update, on 09-16. Elements `_2`, `_5`, `_6` and `_7`
are still running at about three days. Catch-up array `21092481` remains
queued `afterany` behind them. 76 of the 79 new cells are Amazon Beauty, spread
across its whole grid (64 to 34 023) on both tasks. The other three are Synthea TPP
`mambular` cells.

What moved: the classification leader on Amazon Beauty is now
`autoregressive-transformer` at 0.7083. The previous `thp` lead (0.7084) was a
one-seed mean. With all three seeds in, `thp` is at 0.7080, a tie with
`pragma-mlm` and `autoencoder`. Exponents shifted by at most 0.01 (TabGPT
classification 0.093 → 0.095, `mm-tpp` TPP 0.045 → 0.042), and a few
adjacent rows swapped order. None of the four findings in the scale README
changed. `count-logistic` still has only seeds 13 and 29 at Amazon Beauty's
full pool.

### GPU lane cancelled 2026-09-14

Array `21068282` was cancelled to release RTX 3090s. **381 of 576 GPU cells were
already complete and are kept**; 195 remain. Nothing needs recomputing — see
[`gpu_resume.md`](gpu_resume.md) for the state record, what the kept cells
support, and the one-line resubmit. The CPU arrays (`21068281`, `21092481`) were
not touched.

All 18 LM scaling curves (3 datasets x 2 tasks x 3 LM methods) still have enough
points to fit, so the LM entries are reportable now. Two gaps to respect:
`mm-tpp` stops at 65 536 of 93 305 on Synthea and at 8 192 of 34 023 on Amazon
Beauty TPP, and LM seed coverage is thinner than the CPU methods' three seeds.

Two robustness fixes went in alongside the cancel, because a `scancel` is
exactly the case they cover:

* **Result files are now written through a temp file and `os.replace`.**
  `target.open("w")` truncates before the dump, so a worker killed in that window
  left a half-written result. No corruption was found after this cancel, but the
  window was real.
* **`load_results` skips an unreadable result** with a warning instead of
  aborting the whole report. A truncated file also reads as incomplete to
  `--only-missing`, so it gets rerun rather than silently standing.

### Cost

TPP cells expand each sequence into up to eight causal prefixes
(`tpp_max_prefixes_per_user`, default 8), so Synthea's largest TPP cell trains on
roughly 746 000 examples. The prefix count is deliberately **not** reduced at
large `n`: the 624-cell screen used 8, and lowering it here would make the new
curves incomparable with the ones they are meant to extend. Expect on the order
of a few thousand CPU-hours and roughly a week of wall-clock on 32 workers. The
arrays are seven-day, resumable, and requeue before timeout; a restarted worker
skips result files already present.

To finish sooner, raise `--max-array-tasks`, or drop `SCREENING_SEEDS` from three
seeds to one for the two largest sample points.

### Verification before submission

* `load_mbd` reproduces `load_mbd_mini` **exactly** on the same input — identical
  train/validation/test membership, 52 marks, prevalence `0.011440` — so the
  chunked scan, interning and in-scan truncation change no science.
* `_group_rows` returns byte-identical groups to the previous `np.unique`
  implementation over 200 randomised cases.
* One cell per dataset per task runs green, and `slurm/probe_scale_memory.sbatch`
  runs the *largest* cell of each dataset under `--mem=48G` before any array is
  submitted.

## Monitoring

Expand arrays when counting jobs; the normal compressed `squeue` view counts an
entire array as one line.

```bash
squeue -r -u guyhada -o '%i|%j|%T|%M|%R'
```

Count result-level statuses rather than relying only on SLURM exit codes:

```bash
find "$KBC_OUTPUT_ROOT/paper_runs" -mindepth 2 -maxdepth 2 \
  -type f -name '*.json' -print0 \
  | xargs -0 -r -n 200 jq -r '.status // "metadata"' \
  | sort | uniq -c
```
