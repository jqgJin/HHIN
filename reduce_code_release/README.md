# HHIN Approximate Reduction

This repository contains the code and result files used for the HHIN approximate-reduction experiments. All Python entry points are in one directory. Configuration, audited results, and documentation are grouped under `config/`, `results/`, and `docs/`.

## Data

The experiments use the public ACM, DBLP, and IMDB archives from the [Heterogeneous Graph Benchmark](https://www.biendata.xyz/hgb/). The dataset archives are not redistributed here.

Place the files in a local directory such as `data/`:

```text
data/
  ACM.zip
  DBLP.zip
  IMDB.zip
```

The expected SHA-256 values and graph counts are checked by `validate_release.py`.

## Environment

The reported runs used Python 3.10.8 on Windows in CPU mode.

```bash
python -m pip install -r requirements.txt
```

Package versions are pinned in `requirements.txt`. PyTorch installation may require the platform-specific command from the PyTorch distribution page.

## Validate the release

```bash
python validate_release.py --data-dir data --output runs/validation_report.json
```

The validator checks archive hashes, graph counts, DBLP label counts, semantic-budget feasibility, shared original baselines, cosine bounds, and the retained HAN comparison.

## Main commands

Threshold search and semantic-preservation evaluation:

```bash
python hhin_experiments.py --task semantic_select --base-dir data --datasets ACM DBLP IMDB --out-dir runs/semantic_selection
```

The full 10 by 10 threshold grid evaluates all target nodes and is computationally expensive. The settings used in the paper are retained in `config/selected_thresholds.csv`.

Reduction construction cost, architecture control, and DBLP cases:

```bash
python run_reduction_analysis.py --data-dir data --selected-thresholds config/selected_thresholds.csv --output-dir runs/reduction_analysis --repeats 5
```

DBLP KMeans evaluation over all 4,057 public author labels:

```bash
python hhin_experiments.py --task clustering_selected --base-dir data --datasets DBLP --selected-thresholds-csv config/dblp_profiles.csv --out-dir runs/clustering --cluster-algorithms kmeans --sim-methods pathsim,hetesim --seeds 42,43,44,45,46 --kmeans-n-init 10
```

Transductive label-neighborhood evaluation:

```bash
python hhin_experiments.py --task prediction_selected --base-dir data --datasets DBLP --selected-thresholds-csv config/dblp_profiles.csv --out-dir runs/label_neighborhood --sim-methods pathsim,hetesim --prediction-k-values 10
```

Retrieval and ranking runtime:

```bash
python run_query_runtime.py --base-dir data --selected-thresholds-csv config/selected_thresholds.csv --out-root runs/runtime --repeats 10 --warmup-queries 20
```

HGB R-GCN comparison between the original and reduced DBLP main structures:

```bash
python run_hgb_rgcn_adapter.py --data-dir data --output-dir runs/rgcn --seeds 42,43,44,45,46 --epochs 150 --threads 4
```

The adapter uses DGL `RelGraphConv` and the HGB `feats-type=3` identity-feature
protocol. Install `requirements-hgb-rgcn.txt` in a separate environment for a
native DGL/PyTorch combination. On Windows, the script can bypass an unavailable
GraphBolt binary because this full-batch experiment does not use GraphBolt or any
distributed API. The bypass status is recorded in result metadata.

HAN comparison on the same original and reduced DBLP main structures:

```bash
python run_han_validation.py --data-dir data --output-root runs --run-label han --seeds 42,43,44,45,46 --epochs 40 --threads 4
```

The HAN adapter follows the HGB architecture and DBLP model/optimizer
hyperparameters, with a fixed 40-epoch budget for the reported comparison.
It uses the APA and APVPA metapaths in both views; APTPA is excluded because Term
belongs to the strong attribute layer rather than the compared main-structure
carrier. Public HGB author features are used on the original graph, and each
reduced author receives the mean feature vector of its members. Splits and test
authors are aligned with the R-GCN comparison. Each run is written to a new
timestamped directory unless `--allow-overwrite` is given.

Balanced-profile clustering visualization:

```bash
python plot_clustering.py --base-dir data --selected-thresholds-csv config/dblp_profiles.csv --profile Balanced --out-root runs/clustering_figure --paper-figures-dir=
```

## Interpretation boundaries

- Semantic preservation is approximate and evaluated for the specified paths, metrics, and budgets.
- Reduced-graph runtime is measured in compact cluster space. Expansion to complete original-node rankings is excluded.
- Peak RSS is an absolute construction footprint, not a comparative memory-saving result.
- Label-neighborhood evaluation is transductive leave-one-out evaluation, not held-out classifier testing.
- The HGB R-GCN identity-feature experiment is transductive and compares the
  original and reduced main structures under an aligned split.
- The HAN experiment is also transductive and compares only the main structure.
  Reduction and metapath-graph preparation are excluded from per-epoch training time.
- Trace, BSIN, and K-bisimulation values use different protocols and are contextual references rather than a direct ranking.

See `docs/DATA_AUDIT.md` and `docs/RESULT_PROVENANCE.md` for details.

## Release checklist

Before publishing, choose an appropriate software license. The manuscript's Data and Code Availability statement links to https://github.com/jqgJin/HHIN.
