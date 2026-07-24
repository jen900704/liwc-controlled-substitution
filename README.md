# LIWC Controlled Substitution

This release candidate contains a small, auditable implementation of the
controlled-substitution framework from “Disentangling the Interpretive and
Predictive Roles of LIWC: Controlled Substitution in Depression-Related
Classification.”

The framework holds participant-level cross-validation splits and base
representations fixed while replacing the LIWC block:

- **intact** — the observed LIWC feature vector;
- **PCA** — training-fold median imputation, standardization, and PCA, with the
  transformed block padded back to the original width when necessary;
- **shuffled** — complete LIWC rows are permuted within the training fold, while
  test rows are sampled from complete training-fold rows;
- **random-marginal** — each LIWC feature is sampled independently from its
  imputed training-fold empirical marginal.

All fitting operations are fold-local. The classifier pipeline independently
fits median imputation and standardization on each training fold.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## License and permitted use

This repository is publicly available for research transparency and
reproducibility. No open-source software license is granted. Except for rights
provided through GitHub's Terms of Service, permission from the copyright
holders is required to reuse, modify, or redistribute the code.

The repository does not grant rights to any underlying datasets, participant
materials, pretrained models, feature matrices, or proprietary LIWC resources.

## Toy example

From the repository root:

```bash
python scripts/run_primary.py --config configs/toy.json
```

The command writes fold-level scores, draw-averaged scores, condition summaries,
dataset-level contrasts, exact sign-flip p-values, and BH-adjusted q-values to
`outputs/toy/`. The toy data are entirely synthetic.

Category-level interpretation can be exercised separately:

```bash
python scripts/run_interpretability.py \
  --input examples/toy_data.csv \
  --output outputs/toy/category_effects.csv \
  --bootstrap 5000
```

## Expected input schema

The primary command accepts one CSV containing one row per participant-level
record:

| Column | Meaning |
|---|---|
| `dataset` | Dataset block name |
| `record_key` | Unique record key within dataset |
| `label` | Binary outcome encoded as 0 or 1 |
| `base_*` | One or more fixed base-representation features |
| `liwc_*` | One or more numeric LIWC category features |

Column names and prefixes are configurable in JSON. Missing numeric values are
allowed and are imputed from the training fold. Real identifiers are used only
to align authorized local inputs; they are never written to fold outputs.

## Primary and supplementary evaluations

The primary substitution analysis is matched repeated stratified
cross-validation: 10 repetitions of 5 folds (reduced only if the smallest class
cannot support five folds), with the same splits used for every condition.
Shuffled and random-marginal controls may use multiple deterministic draws,
which are averaged within fold before condition and dataset aggregation.

`scripts/run_supplementary_ridge_probe.py` is a separate, explicitly
supplementary single five-fold out-of-fold ridge recoverability probe. It is not
part of the 10 × 5 primary substitution evaluation and its outputs must not be
combined with primary test families.

## Statistical scope

The public core includes dataset-level aggregation, two-sided exact sign-flip
tests, and Benjamini–Hochberg correction. The intended primary family comprises
exactly the eight prespecified dataset-blocked macro-F1 contrasts: for each of
two routes, intact minus base, PCA, shuffled, and random-marginal. Callers are
responsible for defining those routes before applying the eight-test family.

Category-level Cohen’s d uses sample-size-weighted pooled sample variance.
Participants are bootstrapped separately within outcome groups with stable
SHA-256-derived seeds, 5,000 replicates by default, percentile intervals, and
the validated sparse/degenerate interval-unavailability rule.

## Data and reproducibility scope

No restricted datasets, transcripts, audio, clinical metadata, real
participant-level features, fold assignments, LIWC dictionaries, word lists, or
derived participant outputs are distributed. Users must obtain datasets and
LIWC-derived features under their own provider and LIWC licenses; see
`docs/data_access.md`.

This repository reproduces the reusable controlled-substitution algorithms,
matched repeated-CV machinery, aggregation, exact testing, multiplicity
correction, and deterministic category-level effect estimation. It does not
reproduce the paper’s numeric results without authorized datasets,
provider-compliant preprocessing, licensed LIWC outputs, and the paper’s base
representations.

## Citation

Citation details are placeholders pending publication metadata and author
approval. See `CITATION.cff`.
