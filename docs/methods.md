# Implementation notes

## Leakage controls

Each dataset is split at the participant-record level with
`RepeatedStratifiedKFold`. A single split iterator supplies every condition, so
comparisons are matched. PCA, imputation, scaling, shuffling, and empirical
marginal estimation use training rows only. The downstream imputer and scaler
are fitted separately inside each model pipeline.

The PCA component count is
`min(number_of_LIWC_features, max(1, number_of_training_rows - 1))`. Scores are
zero-padded to the original LIWC width when the component count is smaller.
Whitening is disabled.

## Random controls

Participant-shuffled controls preserve complete observed cross-feature row
structure. Training rows are permuted without replacement; test controls sample
training rows with replacement. Random-marginal controls independently sample
each column with replacement from its imputed training-fold marginal, removing
cross-feature dependence.

The primary validated analysis used 30 draws for shuffled and random-marginal
conditions. Draws are averaged within a fold before aggregation. The toy config
uses three draws solely to run quickly.

## Aggregation and inference

Fold scores are averaged within dataset, route, model, and condition. For the
primary dataset-blocked contrasts, intact-minus-comparator differences are
formed within matched dataset-route-model cells and then averaged across the
two validated classifiers inside each dataset block. Exact two-sided sign-flip
tests enumerate all sign assignments to the dataset deltas.

Benjamini–Hochberg correction must be applied to the intended prespecified
family. For the primary table this is exactly eight macro-F1 tests: four
contrasts for each of two routes.

## Interpretation

Cohen’s d is depressed minus non-depressed under the conventional label mapping
`1` minus `0`. Sample variance uses `ddof=1`. Bootstrap resampling is performed
independently within the two participant groups. Stable seeds are derived from
the first eight SHA-256 digest bytes and reduced modulo 2^32.

An interval is unavailable if the observed statistic or interval is nonfinite,
total nonzero support is below two, more than 25% of replicates are undefined,
or a percentile endpoint equals the observed d to absolute tolerance 1e-12.
