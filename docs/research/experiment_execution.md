# Executing the paper experiment programme

The named suites in `eventfm/benchmark/experiments.py` are the executable
counterpart to the paper protocol. A suite fixes datasets, tasks, methods,
sample sizes, seeds, adaptation regimes, and named variants. Each cell writes
one immutable JSON result, so failed jobs can be resumed without rerunning the
matrix.

## 1. Prepare datasets

The primary banking datasets are the default. `--paper` additionally prepares
the four chronological variants and the five public GEM/TPP-LLM datasets.
`--force` is required to replace older processed banking splits with the
train-only preprocessing and natural-evaluation-prevalence protocol.

```bash
python scripts/prepare_data.py --force
python scripts/prepare_data.py --paper --force
```

Every newly prepared dataset records source/citation information, natural and
used prevalence by split, fitted category/quantile artifacts, split strategy,
and SHA-256 hashes of the JSONL splits in `meta.json`.

## 2. Inspect and run suites

```bash
python scripts/run_experiments.py --list-suites
python scripts/run_experiments.py --suite paper-headline --list
python scripts/run_experiments.py --suite paper-headline --array-index 0 --array-size 64
```

For a scheduler array of size `K`, task `i` runs indices `i, i+K, i+2K, ...`.
Results are safe to resume; pass `--overwrite` only when deliberately replacing
a cell. The original `python scripts/run_benchmark.py --all` grid remains
exactly 624 cells and keeps its legacy result keys.

Reusable scheduler templates are provided for CPU and RTX 3090 runs:

```bash
sbatch --export=ALL,KBC_EXPERIMENT_SUITE=objective-factorial \
  slurm/run_experiment_cpu_array.sbatch
sbatch --export=ALL,KBC_EXPERIMENT_SUITE=gem-full-banking \
  slurm/run_experiment_gpu_array.sbatch
```

## 3. Suite blocks and the question each answers

| Block | Suites | Evidence |
| --- | --- | --- |
| Baseline validity | `core-reproduction`, `strong-baselines`, `paper-headline` | Original screen, stronger controls, and the five-seed central table. |
| Objective | `objective-ablation`, `objective-factorial` | Named baselines plus single, pairwise, contrastive, and full joint-loss conditions. |
| Time and TPP | `temporal-models`, `time-representation-ablation` | Recurrent, attentive, intensity-free, convolutional, continuous, bucketed, and rotary time interfaces. |
| Adaptation and representation | `adaptation-regimes`, `pooling-ablation` | Frozen probes, bottleneck-adapter PEFT, full tuning, and customer readout choice. |
| GEM reproduction | `gem-main-banking`, `gem-full-banking`, `gem-reference-datasets`, `gem-template-time-type` | The full 20-tokenizer implementation and the paper's external datasets/template ablation. |
| GEM analysis | `gem-context-budget`, `gem-model-scale`, `gem-event-text` | Equal-event/equal-token budgets, capacity sensitivity, and language-prior isolation. |
| Robustness | `chronological-robustness`, `cross-schema-transfer` | Whole-entity temporal drift and leave-one-dataset-out source pretraining. |
| Scaling | `labeled-adaptation-scaling`, `unlabeled-pretraining-scaling`, `model-scale-ablation`, `context-length-ablation` | Separates labeled data, unlabeled data, model size, and effective history. |

Variants and non-default adaptation regimes are encoded in the result filename,
preventing ablations from overwriting a default cell. Cross-schema runs build a
source-only pretraining pool, a union input vocabulary, and a target-specific
prediction head; the resolved sources are saved in result metadata.

## 4. Aggregate with uncertainty

```bash
python scripts/aggregate_results.py --results-dir /path/to/results
```

The aggregator creates:

- `results.csv`, including fidelity, regime, variant, metrics, and efficiency;
- `results.md`, default full-tuning tables with seed-bootstrap 95% intervals;
- `adaptation_results.md`, frozen/PEFT/full comparisons;
- `ablation_results.md`, every non-default named variant;
- scaling exponent tables and figures, gated by fit quality.

Each new result JSON includes the full training/model/extra configuration,
method reference and disclosed divergence, dataset provenance, git commit and
dirty flag, train/test counts, wall time, trainable and total parameters, and
peak GPU memory when CUDA is used.

## 5. Result gates

A suite being runnable is not evidence for a paper claim. Before a table is
called headline-ready:

1. rebuild datasets with the current converter and archive their hashes;
2. complete every declared seed or retain the recorded failure;
3. select hyperparameters and thresholds on validation only;
4. verify natural validation/test prevalence and train-only fitted artifacts;
5. run both controlled-budget and defensible native-recipe conditions;
6. disclose `Approximation` status in tables and captions;
7. archive per-example predictions when paired entity bootstrap tests are
   required;
8. regenerate all tables from the immutable per-cell JSON files.

## 6. Remaining boundary for a foundation-model claim

The code now implements the comparison matrix, but completed large-scale
pretraining is still an experimental deliverable. A top-tier paper must not
call the current small, in-domain checkpoints a banking foundation model until
the chosen unlabeled corpus, checkpoint scale, transfer tasks, and completed
multi-seed evidence satisfy the gates in `experimental_protocol.md`.
