# Bridging the Decycling-Dismantling Gap: A Structure-Guided Approach for Network Dismantling

This repository contains the code, datasets, pretrained checkpoints, and experiment scripts for the paper **"Bridging the Decycling-Dismantling Gap: A Structure-Guided Approach for Network Dismantling"**.

The main model in this project is **DiffGDM** (Diffusion Graph Dismantling Model), a structure-guided generative framework for network dismantling. DiffGDM is trained on network decycling as a proxy task and, during inference, uses structural guidance to search for node-removal sets that dismantle a network efficiently.

## Overview

Network dismantling aims to find a small set of nodes whose removal collapses the largest connected component of a graph. Classical decycling-based methods first remove a feedback vertex set and then break the resulting forest, but the paper observes that the best decycling set is not always the best dismantling set on real-world networks.

DiffGDM addresses this decycling-dismantling gap by:

- learning structural priors from decycling labels on synthetic graphs;
- using a discrete flow matching / diffusion-style generative model for binary node-selection states;
- injecting structural guidance during inference, such as LCC, global clustering coefficient, average clustering coefficient, global efficiency, natural connectivity, density, or MVC-based cost guidance;
- applying tree-breaking and node-reinsertion post-processing to obtain final dismantling sets;
- supporting independent guided trajectories and ensemble selection across multiple structural metrics.

## Repository Structure

```text
.
|-- model_train/
|   |-- main_SSLtrain_diffusion.py        # Train DiffGDM / diffusion FVS model
|   |-- main_SSLtrain_model_metrics.py    # Train graph metric prediction models used for guidance
|   |-- configs/
|   |   `-- grid_search/                  # Training configuration files
|   |-- dataset/
|   |   `-- train_dataset_new/            # Synthetic training graphs
|   |-- models/                           # GNN, diffusion, and metric model definitions
|   |-- train/                            # Training loops
|   |-- loss_functions/                   # Diffusion and supervised losses
|   |-- optimizers/                       # Optimizer and scheduler wrappers
|   `-- analysis/                         # Training-side analysis and visualization tools
|
|-- model_test/
|   |-- decycling_main.py                 # Decycling experiment entry point
|   |-- dismantle_main.py                 # Dismantling experiment entry point
|   |-- decycling_dismantling_analysis.py # Joint analysis scripts
|   |-- graph_generate.py                 # Synthetic graph generation utilities
|   |-- dataset/
|   |   |-- dataset_synth/                # Synthetic benchmark graphs
|   |   `-- dataset_real_all/             # Real-world benchmark graphs
|   |-- method/
|   |   |-- modelGym/                     # DiffGDM/GDM inference models and checkpoints
|   |   |-- NodeRein/                     # Node reinsertion implementation
|   |   `-- noderein_linux/               # Linux NodeRein helper
|   |-- analysis_visi/                    # Plotting and metric analysis utilities
|   |-- utils/                            # Graph metrics and data utilities
|   `-- result/                           # Saved experiment results
|
|-- requirements.txt                      # Python dependencies
`-- README.md
```

## Environment

The code was developed with Python 3.9/3.10, PyTorch, PyTorch Geometric, NetworkX, and igraph. A CUDA-enabled GPU is recommended for training and large-scale inference.

Example environment:

```bash
conda create -n diffgdm python=3.10
conda activate diffgdm
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

The provided `requirements.txt` follows the experiment environment used by this project and includes additional packages required by the DiffGDM training, inference, graph-analysis, and visualization scripts. It targets the PyTorch 2.0.1 / CUDA 11.8 wheel setup used by the saved environment; if your CUDA or PyTorch version differs, install the matching PyTorch, PyTorch Geometric, and `torch_sparse` wheels for your platform.

Notes:

- The pretrained DiffGDM checkpoints are under `model_test/method/modelGym/trained_models/`.
- The structural metric guidance checkpoints are under `model_test/method/modelGym/Diffusion_model/robustness_metrics_models/`.
- Some baseline algorithms referenced by the experiment scripts, such as MinSum, BPD, CoreHD, DQN, NIRM, CI, FINDER, or DomiRank, depend on external implementations. If you run all baselines, place the corresponding implementations under `model_test/method/` or adjust `method/method.py` to the subset of methods you want to evaluate.

## Data

The repository includes two main data groups:

- `model_train/dataset/train_dataset_new/`: synthetic training graphs generated from Barabasi-Albert, Erdos-Renyi, Watts-Strogatz, and static power-law graph families.
- `model_test/dataset/`: benchmark graphs for evaluation.
  - `dataset_synth/`: synthetic benchmark graphs with different graph families and graph sizes.
  - `dataset_real_all/`: 27 real-world networks used for empirical evaluation.

All graph files are stored in GraphML format.

## Training

Run training commands from the `model_train` directory.

Train the DiffGDM diffusion model:

```bash
cd model_train
python main_SSLtrain_diffusion.py --cfg configs/grid_search/diffusion_model_DFM.yaml --train_mode grid
```

Other diffusion configurations are available in:

```text
configs/grid_search/diffusion_model_DFM.yaml
configs/grid_search/diffusion_model_gaussian.yaml
configs/grid_search/diffusion_model_categorical.yaml
```

Train graph-level structural metric models used for guidance:

```bash
cd model_train
python main_SSLtrain_model_metrics.py --cfg configs/grid_search/model_metrics.yaml --train_mode grid
```

The training scripts support:

- `grid`: grid search over the YAML parameter space;
- `hold_out`: single hold-out validation run;
- `cross_val`: cross-validation run.

Outputs are written to the `out_dir` specified in each YAML configuration.

## Evaluation

Run evaluation commands from the `model_test` directory.

Decycling experiments:

```bash
cd model_test
python decycling_main.py
```

Dismantling experiments:

```bash
cd model_test
python dismantle_main.py
```

The experiment settings are configured near the bottom of each entry script. Common options include:

- `dataset_type`: choose `synth`, `real_all`, or another available dataset suffix;
- `method_list`: choose the evaluated methods or DiffGDM variants;
- `data_cover`: whether to overwrite existing result files;
- `Visual_presentation`: whether to only visualize existing results or recompute them;
- `DiffDGM_infer_param`: guidance settings for DiffGDM inference.

Example DiffGDM inference parameter block:

```python
DiffDGM_infer_param = {
    "gradient_estimator_type": "cost_estimator",
    "guided_model_path": "method/modelGym/Diffusion_model/robustness_metrics_models",
    "use_tag": True,
    "parallel_sampling": 4,
}
```

Supported DiffGDM-style method names include:

- `DiffGDM`: guided DiffGDM using the default MVC/cost guidance;
- `DiffGDM_base`: unguided baseline;
- `DiffGDM_LCC`, `DiffGDM_global_CC`, `DiffGDM_average_CC`, `DiffGDM_global_efficiency`, `DiffGDM_natural_connectivity`, `DiffGDM_density`: metric-specific guided variants, when the corresponding guidance model is available;
- `CoreDiffGDM` and `CoreDiffGDM_base`: CoreDiffGDM variants used in the dismantling scripts.


## Citation

If you use this code or the DiffGDM model, please cite:

```bibtex
@misc{zhang2026bridging,
  title = {Bridging the Decycling-Dismantling Gap: A Structure-Guided Approach for Network Dismantling},
  author = {Zhang, Peng and Fu, Jun and Cheng, Tonglei and Findeisen, Rolf},
  year = {2026},
  note = {Manuscript}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
