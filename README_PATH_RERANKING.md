# InfoVQA Path-Aware Reranking

The default MEHR candidate selection is unchanged. Path-aware reranking is
opt-in and reranks the cached MEHR candidate facts with
`alpha * fact_score + beta * tile_score + gamma * root_score`.

Run the test ablation on `QAs_test.json`:

```bash
PYTHONPATH=. python3 -m src.experiment.experiments.infovqa_ablation \
  --root-k 100 --tile-k 100 --final-k 10 \
  --normalization raw --path-reranking
```

The script uses equal fact/tile/root weights by default. Pass
`--weights-file` to use a different full-path weight configuration. It writes
`path_reranking_test_summary.json`,
`path_reranking_per_query.jsonl`, `path_reranking_paired_analysis.json`, and
`path_score_distributions.json`.
