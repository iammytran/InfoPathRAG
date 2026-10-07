# InfoPathRAG Path-Aware Reranking

The default MEHR candidate selection is unchanged. Path-aware reranking is
opt-in and reranks the cached MEHR candidate facts with
`alpha * fact_score + beta * tile_score + gamma * root_score`.

Run the test ablation on `QAs_test.json`:

```bash
PYTHONPATH=. python3 -m src.experiment.experiments.infopathrag_ablation \
  --ablation-type reranking \
  --root-k 100 --tile-k 100 --final-k 10 \
  --normalization raw
```

The script uses equal fact/tile/root weights by default. Pass
`--weights-file` to use a different full-path weight configuration. It writes
`path_reranking_test_summary.json`,
`path_reranking_per_query.jsonl`, and
`path_score_distributions.json`.

## Hyperparameter sensitivity

Run the weight and top-k sensitivity study with both raw and query z-score
path scores:

```bash
PYTHONPATH=. python3 -m src.experiment.experiments.infopathrag_hyperparameter_sensitivity
```

To run the same configurations using raw scores only:

```bash
PYTHONPATH=. python3 -m src.experiment.experiments.infovqa_hyperparameter_sensitivity_raw
```

To run the same configurations using query z-score normalization only:

```bash
PYTHONPATH=. python3 -m src.experiment.experiments.infovqa_hyperparameter_sensitivity_zscore
```

The raw-only script writes to
`algorithm_results/InfoPathRAG/InfoVQA/hyperparameter_sensitivity_raw` by
default, while the z-score-only script writes to
`algorithm_results/InfoPathRAG/InfoVQA/hyperparameter_sensitivity_zscore`.
Sensitivity plots label each configuration with its actual
`(fact, tile, root)` weights or `(root_k, tile_k, final_k)` top-k tuple.
