# InfoVQA Path-Aware Reranking

The default MEHR candidate selection is unchanged. Path-aware reranking is
opt-in and reranks the cached MEHR candidate facts with
`alpha * fact_score + beta * tile_score + gamma * root_score`.

Run the validation weight grid:

```bash
PYTHONPATH=. python3 -m src.experiment.experiments.infovqa_ablation \
  --split validation \
  --qa-path datasets/InfoVQA/QAs_val.json \
  --root-k 100 --tile-k 100 --final-k 10 \
  --normalization raw --path-reranking
```

Then run the locked test evaluation:

```bash
PYTHONPATH=. python3 -m src.experiment.experiments.infovqa_ablation \
  --split test \
  --qa-path datasets/InfoVQA/QAs_test.json \
  --root-k 100 --tile-k 100 --final-k 10 \
  --weights-file algorithm_results/LILaC/InfoVQA/ablation/selected_path_weights.json \
  --normalization raw --path-reranking
```

Validation writes `path_weight_search_validation.json` and
`selected_path_weights.json`. Test writes `path_reranking_test_summary.json`,
`path_reranking_per_query.jsonl`, `path_reranking_paired_analysis.json`, and
`path_score_distributions.json`.
