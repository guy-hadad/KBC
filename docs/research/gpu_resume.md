# Resuming the cancelled GPU lane

The `scale-datasets` GPU array (`21068282`, the three LM entries) was cancelled
on 2026-09-14 to release RTX 3090s. **Nothing was lost.** Completed cells are
JSON files on group storage and survive a cancel; only cells that were mid-run
at `scancel` are gone, and those had written nothing.

## State at cancellation

| | Cells |
|---|---:|
| GPU cells in `scale-datasets` | 576 |
| Completed `ok` and kept | **381** |
| Remaining | 195 |

Machine-readable record, including the full list of remaining cell keys:

```
$KBC_OUTPUT_ROOT/paper_runs/resume/gpu_scale-datasets.json
```

Per method: `tpp-llm` 134, `language-tpp` 127, `mm-tpp` 120.

## The kept results are usable as they stand

All **18** of the (dataset x task x LM method) scaling curves have enough points
to fit `E(N) = a · N^-b`, which needs three:

| Dataset | Points per LM curve | Reaches the full pool? |
| --- | ---: | --- |
| MBD (full) | 9-10 | yes, all six curves |
| Synthea EHR | 10-12 | yes, except `mm-tpp` (tops out at 65 536 of 93 305) |
| Amazon Beauty 2014 | 6-8 | yes, except `mm-tpp` on TPP (tops out at 8 192 of 34 023) |

So the LM entries can be reported now, with two caveats: `mm-tpp` is short of its
largest point on Synthea and on Amazon Beauty TPP, and seed coverage is thinner
than the three seeds the CPU methods have. Check `results.csv` for the seed count
behind any LM number before putting it in a table.

## Resubmitting

The gap-filling mode makes this a single command — it reruns only cells with no
result or a result that is not `ok`, so the 381 kept cells are never recomputed:

```bash
sbatch --array=0-15 \
  --export=ALL,KBC_EXPERIMENT_SUITE=scale-datasets,KBC_ONLY_MISSING=1 \
  slurm/run_experiment_gpu_array.sbatch
```

Check what it would do first:

```bash
python scripts/run_experiments.py --suite scale-datasets --only-gpu-only \
  --only-missing --list | tail -1
```

No prerequisite gate is needed — the datasets are prepared and the subset cache
is materialised. Resubmit whenever GPU frees up; the arrays are seven-day and
resumable, so a partial run simply shortens the next one.
