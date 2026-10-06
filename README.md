# InfoPathRAG

InfoPathRAG explores retrieval from infographic documents. This repository includes the LILaC baseline used in our experiments.

## Run InfoPathRAG
1. Processing images to tiles + serialize & embed image/tiles
python -m src.lilac.lcg_constructor.estimate_components
python /Users/mytnguyen/Documents/InfoPathRAG/src/lilac/lcg_constructor/tile_infographics.py

2. Embed query
./scripts/query_decomposition/query_decomposer.sh                           
./scripts/query_decomposition/modality_estimator.sh                 
./scripts/query_decomposition/queryset_embedder.sh --embedder mmembed 

3. Retrieve


4. Generation


4. Hyperparameter sensitivity



## LILaC baseline

The baseline code is forked from the [original LILaC repository](https://github.com/joohyung00/lilac).

### Requirements

Our experiments were run on four NVIDIA RTX 3090 GPUs, each with 24 GB of VRAM.

### Reproduce the baseline

From the repository root, run:

```bash
make
```

### Use precomputed artifacts

To skip the full baseline run, download the precomputed artifacts from **[link](https://drive.google.com/file/d/1QBL1ehw2DEodP6lJONqLKi3SLT_Iy09t/view?usp=sharing)** and extract them into the locations expected by the repository's Makefile. Then run:

```bash
make retrieve
make visualize
```

`make retrieve` runs retrieval using the available artifacts; `make visualize` generates visualizations of the baseline results.
