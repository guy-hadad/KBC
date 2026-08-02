# EventFM — benchmarking temporal-semantic foundation models for event streams

This repository turns the survey in `KBC__TPP.pdf` ("Toward a Temporal-Semantic
Foundation Model for Event Prediction") into a runnable comparison: **every
architecture the note discusses, evaluated on every open benchmark it names, on
both of the tasks it cares about.**

* **What** — 12 architectures plus 2 non-neural controls.
* **Where** — the four open benchmarks from §3.1.5: BankSim, PaySim,
  IBM AML (HI-Small) and MBD-mini.
* **How measured** — binary sequence **classification** (a forward-looking
  business label) and a marked **temporal point process** task (predict the next
  event type *and* the time until it happens).
* **Scaling** — every cell is run at six training-set sizes, so each method gets
  its own sample-scaling curve rather than one number.

Everything is built on the Hugging Face ecosystem: models subclass
`PreTrainedModel`, training goes through `Trainer`/`TrainingArguments`, the LM
entries use `AutoModel` + `peft` LoRA, Mamba comes from `transformers`, and the
datasets are pulled with `huggingface_hub`.

The reasoning behind the dataset construction, the label definitions and the
fairness controls is in **[docs/benchmark_design.md](docs/benchmark_design.md)** —
read that before trusting any number here. The paper-facing specification of
where this benchmark still falls short of a publishable result lives in
**[docs/research/](docs/research/)**.

This is an **architecture-screening benchmark**, not a foundation-model result:
the training sets are in the low thousands of sequences, several named methods
are controlled approximations, and pretraining shares the same small corpus as
downstream training.

---

## Results

Generated tables and figures live in **[docs/benchmark/](docs/benchmark/)**:

| File | What it holds |
| --- | --- |
| [`results.md`](docs/benchmark/results.md) | all methods per dataset and task, both tasks |
| [`results.csv`](docs/benchmark/results.csv) | every cell: metric × method × dataset × sample size |
| `figures/by_method/` | **one scaling curve per method**, its four datasets overlaid |
| `figures/by_dataset/` | per dataset, small multiples faceted by architecture family |

Figure filenames are deterministic:

```text
figures/by_method/<method>__<task>__<metric>.svg      # + .dark.svg
figures/by_dataset/<dataset>__<task>__<metric>.svg    # + .dark.svg
```

with `<metric>` one of `auc`, `average_precision`, `accuracy`, `macro_f1`
(classification) or `next_type_accuracy`, `next_type_macro_f1`,
`delta_log_rmse`, `delta_log_mae` (TPP). Each figure ships a light and a dark
variant.

### Method scaling curves, at a glance

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmark/figures/by_dataset/ibm_aml__classification__auc.dark.svg">
  <img src="docs/benchmark/figures/by_dataset/ibm_aml__classification__auc.svg" alt="IBM AML classification ROC-AUC against training-set size, faceted by architecture family">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmark/figures/by_dataset/banksim__tpp__next_type_macro_f1.dark.svg">
  <img src="docs/benchmark/figures/by_dataset/banksim__tpp__next_type_macro_f1.svg" alt="BankSim next-event-type macro F1 against training-set size, faceted by architecture family">
</picture>

---

## The method zoo

> **Fidelity matters here.** `Implemented` marks the two controls, which have no
> paper to be faithful to. Every other entry is an `Approximation`: it preserves
> the central comparison but differs materially from the cited recipe, usually
> in scale or in the pretraining corpus. Each one declares its divergence in the
> registry, which is enforced by a test and surfaced in `results.csv`. **A row
> named after a paper is not a reproduction of that paper** — see
> [docs/research/method_catalog.md](docs/research/method_catalog.md).

| Key | Paper / system | Family | Fidelity |
| --- | --- | --- | --- |
| `count-logistic` | non-neural control | classic | Implemented |
| `markov` | non-neural control | classic | Implemented |
| `ntpp-gru` | Shchur et al., *Neural temporal point processes: a review* | neural TPP | Approximation |
| `thp` | Transformer-Hawkes-style attention intensity model | neural TPP | Approximation |
| `pragma` | Ostroukhov et al., *PRAGMA: Revolut foundation model* | hierarchical | Approximation |
| `pragma-mlm` | PRAGMA with its masked-value pretraining stage | hierarchical | Approximation |
| `tabbert` | Padhi et al., *Tabular transformers* (BERT variant) | tabular transformer | Approximation |
| `tabgpt` | Padhi et al., *Tabular transformers* (GPT variant) | tabular transformer | Approximation |
| `nvidia-tfm` | NVIDIA Transaction Foundation Model blueprint | tabular transformer | Approximation |
| `coles` | Babaev et al., *CoLES* (pytorch-lifestream recipe) | contrastive | Approximation |
| `mambular` | Thielmann et al., *Mambular* | state space | Approximation |
| `tpp-llm` | Liu & Quan, *TPP-LLM* | LLM | Approximation |
| `language-tpp` | Kong et al., *Byte-token enhanced LMs for TPP analysis* | LLM | Approximation |
| `mm-tpp` | Li et al., *Long-range modeling of multimodal event sequences* | LLM | Approximation |

All non-hierarchical models share one event embedding and one pair of task
heads, and differ **only** in the sequence mixer and the pretraining objective —
that is what makes the table a comparison of architectures rather than of
implementations. PRAGMA keeps its own hierarchical encoder, because the
hierarchy is the method.

The three LM entries share one frozen base LM with LoRA adapters and differ only
in how the history is serialised — textual marks with a continuous temporal
embedding (TPP-LLM), byte-token inter-arrival times in the prompt
(Language-TPP), or byte-token times plus temporal-similarity compression
(MM-TPP). That isolates the tokenisation question the project note asks.

## The datasets

| Dataset | Source | Sequence = | Mark | Marks |
| --- | --- | --- | --- | --- |
| BankSim | `github:atavci/fraud-detection-on-banksim-data` | customer | merchant category | 15 |
| PaySim | `hf:theman10/paysim` | destination account | transfer type | 4 |
| IBM AML (HI-Small) | `hf:OsamaMIT/IBM-AML-HI-Small` | receiving account | payment format | 7 |
| MBD-mini | `hf:ai-lab/MBD-mini` | bank client | transaction event type | 54 |

Raw downloads, converted splits and run outputs are kept off the git checkout on
group storage. Override with environment variables:

```bash
export KBC_DATA_ROOT=/groups/bshapira_group/guyhada/KBC_data
export KBC_OUTPUT_ROOT=/groups/bshapira_group/guyhada/KBC_outputs
export HF_HOME=/groups/bshapira_group/guyhada/hf_cache
```

---

## Install

```bash
conda activate KBC       # or: python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Reproduce

```bash
# 1. download the four benchmarks and convert them to the shared JSONL contract
python scripts/prepare_data.py

# 2. one cell, to check the setup
python scripts/run_benchmark.py --dataset banksim --task tpp --method thp --samples 512

# 3. the full 624-cell grid as a SLURM GPU array
sbatch slurm/run_benchmark_array.sbatch

# 4. tables and scaling figures
python scripts/aggregate_results.py
```

`scripts/run_benchmark.py --all --list` prints the grid. `--skip-gpu-only`
excludes the three LM entries for a CPU-only run.

---

## Data contract

One JSON object per entity, one line each:

```json
{
  "user_id": "C1093826151",
  "events": [
    {"event_type": "es_transportation", "timestamp": 1640995200.0,
     "features": {"amount": "b_3", "merchant": "M348934600"}},
    {"event_type": "es_health", "timestamp": 1641081600.0,
     "features": {"amount": "b_11", "merchant": "M1823072687"}}
  ],
  "label": 1,
  "profile": {"age": "4", "gender": "M"}
}
```

Adding a benchmark means writing one adapter that returns a `TransactionTable`
(`eventfm/datasets/adapters.py`) and registering it; bucketing, grouping,
labelling, splitting and vocabulary construction are shared.

Adding a method means subclassing `FlatMethod` and setting a backbone, or
implementing `Method.run` directly, then calling `register_method`
(`eventfm/methods/`). Both tasks and every dataset come for free.

## Repository layout

```text
configs/                 Hydra config groups (legacy single-run entrypoints)
eventfm/data/            schema, tokenizer, flat + hierarchical + text collators
eventfm/datasets/        the four benchmark adapters and shared conversion code
eventfm/models/          HF-compatible backbones and task heads
eventfm/methods/         the method registry: one module per paper family
eventfm/benchmark/       grid runner, result aggregation, tables and figures
scripts/                 prepare_data / run_benchmark / aggregate_results
slurm/                   array driver and GPU smoke test
docs/                    benchmark design notes and generated report
tests/                   smoke tests
```

## Legacy single-run entrypoints

`docs/scaling_laws/` holds the earlier synthetic-only sweep (PRAGMA vs. two
controls, generated data). It is superseded by `docs/benchmark/` and kept only
for reference.

The original Hydra-driven scripts still work for the synthetic dataset and for
training a single PRAGMA checkpoint:

```bash
python scripts/make_synthetic.py
python scripts/train.py task=pretrain
python scripts/train.py task=classification task.checkpoint_path=outputs/pretrain
python scripts/train.py task=tpp task.checkpoint_path=outputs/pretrain
```

## Tests

```bash
pytest
python -m py_compile $(find eventfm scripts -name "*.py")
```
