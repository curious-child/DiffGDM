# Reproducing the paper figures

Generate the current main-text and supplementary plots from locally saved final
results. The entry calls the original plotting functions; CUDA and an experiment
server are not needed for plotting.

## Setup

Use Python 3.10. From the repository root:

```bash
conda create -n diffgdm-figures python=3.10
conda activate diffgdm-figures
pip install -r experiments_figure/requirements.txt
git lfs install
git lfs pull
```

The graph-level ensemble endpoints use Git LFS. After checkout and `git lfs pull`,
all plotting inputs are local.

## Commands

```bash
python experiments_figure/generate_figures.py --paper-figures 1 2 3 4 6 7 8 S1 S2 S3 S4 --check
python experiments_figure/generate_figures.py --paper-figures 2 3 8 S1 S2 S3
python experiments_figure/generate_figures.py --paper-figures 1 2 3 4 6 7 8 S1 S2 S3 S4
python experiments_figure/generate_figures.py --paper-figures 7 S4 --overwrite
python experiments_figure/generate_figures.py --paper-figures 8 --output-dir ./my_figures
```

`--check` verifies source files and checksums without writing outputs.
The default output directory is `experiments_figure/outputs/paper/`. Each figure
has a PDF and 300-dpi PNG; the framework diagram has a 2400-pixel preview.
`run_manifest.json` records output checksums.
Main-text files use names such as `Figure02`; supplements use `FigureS2`.
The script also works from another working directory using its full path.
The Bash wrapper accepts the same options and `--python /path/to/python`.

## Figure-to-source map

Paths in the data column are relative to `experiments_figure/data/`.

| Figure | Content | Data | Original plotting function |
|---|---|---|---|
| 1 | Framework diagram | `assets/model_framework.pdf` | Original vector asset, with a PNG preview |
| 2 | Synthetic comparisons | `figure2.json` | `LCC_size_removed_nodes.draw_verified_synthetic_comparison` |
| 3 | Real-network comparisons | `figure3.json` | `LCC_size_removed_nodes.draw_verified_route_comparison` |
| 4 | Structural-guidance AUC | `auc/` | `decycling_dismantling_analysis.plt_graph_auc_analysis` |
| 5 | Toy network | Original PDF: `reference/Figure5.pdf`; experiment: `model_test/toy_model_analysis.py` | `network_dismantling_topology.draw_dimantle_nodes` |
| 6 | Structural-guidance comparison | `figure6.json` | `LCC_size_removed_nodes.draw_verified_guidance_comparison` |
| 7 | Forest counts and CFN changes | `residual_forest.json` | `LCC_size_removed_nodes.draw_guidance_benefit_analysis` |
| 8 | Runtime and relative CFN | `figure8.json`, `records/` | `LCC_size_removed_nodes.draw_relative_resource_comparison` |
| S1 | Training and validation loss | `training/figure_s1_record_scores.json` | `data_visualization.model_evaluation_metrics_curves` |
| S2 | Rewrite rounds | `parameters/dismantle_perform_Rewrite_Steps.json` | `dismantle_model_params_analysis.draw_paper_parameter_sensitivity` |
| S3 | Four hyperparameter sweeps | Other four JSON files in `parameters/` | Same parameter-sensitivity function |
| S4 | Forest metric changes | `residual_forest.json` | Same paired forest plotting function as Figure 7 |

Figure 5's original PDF and experiment code are included. The original script
saved the six-panel image, but did not serialize its selected node sets. The
recovered entry below can save those sets as JSON when the experiment is run.
The all-figure command generates the other eleven figures; Figure 5 uses its
dedicated experiment entry. Full rerun agreement with the historical panels
has not yet been verified.
The `reference/` PDFs support visual comparisons; the experimental plots are
generated from numerical data, rather than from these reference PDFs.

## Data and statistics

- Figures 2, 3 and 6 use the final single-run comparisons, including the added
  Random/T2T controls. Older ten-seed studies remain separate.
- S2/S3 directly average 100 saved graphs from each of BA, ER, WS and PL, with
  equal family weights. Only means are shown, without resampling or shaded bands.
- S1 uses the original 400-epoch trace from `DFM_training_test`, configuration 1.
  The saved curves were checked against the original vector figure. The training
  configuration is included alongside the trace.
- Figure 8 has ten graph instances per scale. Relative CFN divides each method's
  mean CFN by the Random-Decycle mean at that scale. Original CFNs, random
  denominators, runtime series and the timing definition are retained.
- Figures 7 and S4 share the same 27 networks and forest-statistics cache.

`data/manifest.json` maps figures to inputs and records repository-relative data
and plotting-source checksums. `records/` retains graph-level control/ensemble
endpoints. Existing `model_test/result/` data remain in place. Numerical input
changes require an intentional corresponding update of the manifest.

## Running experiments

Figure generation only reads saved results. The original experiment entries are:

| Task | Entry, relative to the repository root |
|---|---|
| Decycling-model training | `model_train/main_SSLtrain_diffusion.py` |
| Structural-metric predictor training | `model_train/main_SSLtrain_model_metrics.py` |
| Decycling evaluation | `model_test/decycling_main.py` |
| Dismantling evaluation | `model_test/dismantle_main.py` |
| Toy-network experiment (Figure 5) | `model_test/toy_model_analysis.py` |
| Hyperparameter evaluation | `model_test/dismantle_model_params_analysis.py` |

Use the original configuration blocks and the experiment environment described in
the main README. Hyperparameter sensitivity explicitly evaluates direct ranked
removal followed by reinsertion, separately from decycling-based evaluations.
Training, inference and external baselines require additional dependencies beyond
the plotting environment.

### Figure 5: Toy network

The recovered experiment uses a connected caveman graph with 20 cliques of four
nodes, graph-generation seed 19 and the original `target=10` (10%) setting.
It evaluates DiffGDM_base, MVC-guided DiffGDM and BPD, then calls the original
node-highlighting function for the six panels. The old `CoreDiffGDM` method names
are mapped to the current public names, retaining decycling-based postprocessing.
Model functions retain their own existing inference-seed settings.

From the repository root:

```bash
# Check network generation only; no model or baseline is loaded.
python model_test/toy_model_analysis.py --check

# Run the Toy experiment using the full experiment environment.
python model_test/toy_model_analysis.py

# Redraw its saved node sets without running algorithms again.
python model_test/toy_model_analysis.py --plot-only model_test/result/DiffGDM/toy_model/Figure05.json --output-dir ./toy_preview
```

Outputs are `Figure05.pdf`, `Figure05.png` and `Figure05.json` under
`model_test/result/DiffGDM/toy_model/`. The JSON contains the graph edges,
method order, configuration and both selected node sets for each method.
Existing outputs require `--overwrite`; `--show` opens the figure window.
The archived `19.png` and manuscript PDFs are not overwritten. The recovered
entry has been checked for graph generation and API compatibility; a complete
algorithm rerun against the historical six panels has not yet been performed.

The legacy `--figures`, `--all` and `--profile` options remain available.
Use `--paper-figures`/`--paper-all` for the portable current-paper version.
Exit code 0 means success, 3 identifies an unavailable source item, and 1 indicates
a data or rendering error.
