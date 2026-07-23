# HHIN Approximate Reduction

This repository contains the code used for the HHIN approximate-reduction experiments. All Python entry points are in one directory, and the paper settings are grouped under `config/`. Each command writes its results to the output directory supplied by the user.

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
python validate_release.py --data-dir data --skip-results --output runs/validation_report.json
```

This command checks archive hashes, graph counts, and DBLP label counts. If an
audited result bundle is available, omit `--skip-results` and pass its directory
with `--results-dir`; the validator then also checks semantic-budget feasibility,
shared original baselines, cosine bounds, and the retained HAN comparison.

## Main commands

Threshold search and semantic-preservation evaluation:

```bash
python hhin_experiments.py --task semantic_select --base-dir data --datasets ACM DBLP IMDB --out-dir runs/semantic_selection
```

The full 10 by 10 threshold grid evaluates all target nodes and is computationally expensive. The settings used in the paper are retained in `config/selected_thresholds.csv`.
Attached-feature cosines are evaluated in double precision. At the strict boundary
`tau_x = 1.0`, a `1e-10` comparison tolerance absorbs floating-point roundoff
without relaxing the threshold.

Reduction construction cost, architecture control, and DBLP cases:

```bash
python run_reduction_analysis.py --data-dir data --selected-thresholds config/selected_thresholds.csv --output-dir runs/reduction_analysis --repeats 5
```

DBLP KMeans evaluation over all 4,057 public author labels:

```bash
python hhin_experiments.py --task clustering_selected --base-dir data --datasets DBLP --selected-thresholds-csv config/dblp_profiles.csv --out-dir runs/clustering --cluster-algorithms kmeans --sim-methods pathsim,hetesim --seeds 42,43,44,45,46 --kmeans-n-init 10
```

Similarity-neighborhood label-consistency evaluation:

```bash
python hhin_experiments.py --task prediction_selected --base-dir data --datasets DBLP --selected-thresholds-csv config/dblp_profiles.csv --out-dir runs/label_neighborhood --sim-methods pathsim,hetesim --prediction-k-values 10
```

Cross-dataset architecture control and Guard-component ablation:

```bash
python run_cross_dataset_layer_ablation.py --data-dir data --selected-thresholds config/selected_thresholds.csv --output-dir runs/cross_dataset_layer_ablation
```

For ACM, DBLP, and IMDB, the script compares a single-layer mixed HIN control
with the HHIN layered reduction and evaluates structure only, Term Guard only,
attached-feature Guard only, and both Guards under the same fixed-point candidate
partition and selected thresholds. It also records the candidate pairs uniquely
rejected by each Guard, archive hashes, package versions, and the exact threshold
configuration.

DBLP downstream ablation of the two strong-attribute channels:

```bash
python run_guard_component_ablation.py --data-dir data --output-dir runs/guard_component_ablation --stage non_gnn --tau-h 0.4 --tau-x 1.0 --seeds 42,43,44,45,46
```

This experiment holds the DBLP main-structure candidate partition fixed at the
Balanced profile and evaluates structure only, the high-value attribute-node
Guard, the attached-feature Guard, and both Guards with the same labeled authors,
semantic paths, original-fitted standardization, KMeans settings, and random
seeds. The output contains per-seed NMI, summary statistics, reduction ratios,
and semantic-preservation diagnostics. KMeans uses the Lloyd algorithm with a
single BLAS thread so that the fixed seeds reproduce the reported values across
repeated runs on the same software stack.

Optional DBLP attribute-conflict diagnostic:

```bash
python run_attribute_guard_analysis.py --data-dir data --output-dir runs/attribute_guard_diagnostic --tau-h 0.4 --random-runs 200 --random-seed 20260721 --skip-semantics
```

This matched-random diagnostic is supplementary to the component and
architecture controls reported in the revised manuscript.

Retrieval and ranking preservation at the selected thresholds:

```bash
python hhin_experiments.py --task retrieval_ranking_selected --base-dir data --datasets ACM DBLP IMDB --selected-thresholds-csv config/selected_thresholds.csv --out-dir runs/retrieval_ranking
```

Candidate rankings use similarity in descending order and original node
identifier in ascending order at an exact tie. NDCG uses the mean gain within
each tied predicted-score group, and the output includes both deterministic
top-k overlap and tie-aware overlap. The regression tests for these rules are:

```bash
python -m unittest -v test_ranking_ties.py
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
It uses APA coauthorship and APVPA venue-mediated metapaths in both views. Both
are Author-to-Author paths contained in the main structure; APTPA is excluded
because it traverses Term in the strong attribute layer and would change the
compared carrier and timing scope. Public HGB author features are used on the original graph, and each
reduced author receives the mean feature vector of its members. Splits and test
authors are aligned with the R-GCN comparison. Each run is written to a new
timestamped directory unless `--allow-overwrite` is given.

Optional five-view HAN component diagnostic:

```bash
python run_guard_han_ablation.py --data-dir data --output-dir runs/guard_han_ablation --tau-h 0.4 --tau-x 1.0 --seeds 42,43,44,45,46 --epochs 40 --threads 4
```

This exploratory run evaluates the original, structure-only, Term-only,
attribute-only, and dual-Guard views. It is retained for reproducibility but is
not used as the primary Attribute-Guard evidence because the Term-only versus
dual-Guard HAN differences are not statistically significant.

Balanced-profile clustering visualization:

```bash
python plot_clustering.py --base-dir data --selected-thresholds-csv config/dblp_profiles.csv --profile Balanced --out-root runs/clustering_figure --paper-figures-dir=
```

Thresholds can be reselected from a completed grid without recomputing the
similarity matrices:

```bash
python reselect_thresholds.py --summary-csv runs/semantic_selection/semantic_selection_summary.csv --out-dir runs/reselection
```

`plot_paper_results.py` regenerates the reduction, preservation, and retrieval
figures from audited summary CSV files. Run `python plot_paper_results.py --help`
for the required input paths.

## Interpretation boundaries

- Semantic preservation is approximate and evaluated for the specified paths, metrics, and budgets.
- Reduced-graph runtime is measured in compact cluster space. Expansion to complete original-node rankings is excluded.
- Peak RSS is an absolute construction footprint, not a comparative memory-saving result.
- Similarity-neighborhood label consistency is a transductive leave-one-out diagnostic, not held-out classifier testing; R-GCN and HAN provide the supervised node-classification evidence.
- The cross-dataset component studies hold the candidate partition and thresholds
  fixed within each dataset. They identify dataset-dependent contributions of
  the Term and attached-feature Guards and do not claim universal dominance by
  either channel.
- The single-layer mixed HIN is an architecture control, not a competing
  reduction algorithm. Its purpose is to isolate the effect of representing
  structural and descriptive semantics in separate HHIN layers.
- The HGB R-GCN identity-feature experiment is transductive and compares the
  original and reduced main structures under an aligned split.
- The HAN experiment is also transductive and compares only the main structure.
  Reduction and metapath-graph preparation are excluded from per-epoch training time.
- Trace, BSIN, and K-bisimulation values use different protocols and are contextual references rather than a direct ranking.
