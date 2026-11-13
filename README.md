# Bridging the Decycling-Dismantling Gap: A Structure-Guided Approach for Network Dismantling

Implementation of **Bridging the Decycling-Dismantling Gap: A Structure-Guided Approach for Network Dismantling** by Peng Zhang, Jun Fu and Rolf Findeisen.

DiffGDM learns a generative prior from feedback vertex sets and uses structural guidance to refine node selections for network dismantling. Generated scores support two downstream strategies: decycling followed by tree breaking and node reinsertion, or direct node removal followed by reinsertion. The ensemble variant combines structural guides and selects the smallest feasible dismantling set.

This repository contains model training and evaluation code, pretrained checkpoints, network datasets and saved results for the manuscript figures.

## Installation

Use Python 3.10. The experiment environment uses PyTorch 2.0.1 with CUDA 11.8, PyTorch Geometric, NetworkX and igraph.

```bash
git clone https://github.com/curious-child/DiffGDM.git
cd DiffGDM
git lfs install
git lfs pull

conda create -n diffgdm python=3.10 -y
conda activate diffgdm
python -m pip install -r requirements.txt
```

For another CUDA version, install compatible PyTorch and PyTorch Geometric packages. To run the baseline experiments, install their third-party implementations and native tools under `model_test/method/`, including BPD and MinSum. These dependencies are separate from the saved-result plotting environment.

For plotting saved results, use the smaller environment instead:

```bash
conda create -n diffgdm-figures python=3.10 -y
conda activate diffgdm-figures
python -m pip install -r experiments_figure/requirements.txt
```

## Data and checkpoints

```text
model_train/dataset/                         Synthetic training graphs
model_test/dataset/                          Evaluation networks
model_test/method/modelGym/trained_models/   Generative-model checkpoints
model_test/method/modelGym/Diffusion_model/
  robustness_metrics_models/                Structural-guidance predictors
model_test/result/DiffGDM/                   Saved experiment results
experiments_figure/data/                     Plotting inputs and source records
```

Synthetic experiments use Barabási–Albert, Erdős–Rényi, Watts–Strogatz and power-law networks. Real-network evaluations cover 27 networks. Graphs are stored in GraphML format. Large ensemble source records use Git LFS; `git lfs pull` retrieves them into the local directory structure.

## Reproduce manuscript figures

From the repository root, reproduce the manuscript figures:

```bash
python experiments_figure/generate_figures.py --paper-all
```

The command uses the original plotting and experiment entries and writes PDF and PNG files to `experiments_figure/outputs/paper/`. Saved-result plots read local data; the Toy comparison runs its original algorithms in the experiment environment. Select any figure by its manuscript number:

```bash
python experiments_figure/generate_figures.py --paper-figures 2 3
python experiments_figure/generate_figures.py --paper-figures 5
python experiments_figure/generate_figures.py --paper-figures S2 S3 --overwrite
```

The [figure guide](experiments_figure/README.md) lists the data sources, plotting functions and reproduction commands for all main-text and supplementary figures.

## Training

Run the training scripts from `model_train/`:

```bash
cd model_train
python main_SSLtrain_diffusion.py --cfg configs/grid_search/diffusion_model_DFM.yaml --train_mode grid
python main_SSLtrain_model_metrics.py --cfg configs/grid_search/model_metrics.yaml --train_mode grid
```

The first command trains the generative model; the second trains structural-metric predictors. YAML files specify the architecture, dataset, optimization settings and output directory. The scripts support `grid`, `hold_out` and `cross_val` modes.

## Evaluation

Run the evaluation scripts from `model_test/`:

```bash
cd model_test
python decycling_main.py
python dismantle_main.py
```

Select the dataset, methods and inference settings in each script's configuration block. `Visual_presentation` selects visualization of saved results; `data_cover` controls replacement of existing results. Guidance settings are passed through `DiffDGM_infer_param`, for example:

```python
DiffDGM_infer_param = {
    "gradient_estimator_type": "cost_estimator",
    "guided_model_path": "method/modelGym/Diffusion_model/robustness_metrics_models",
    "use_tag": True,
    "parallel_sampling": 4,
}
```

`DiffGDM` uses MVC guidance, `DiffGDM_base` disables rewrite, and `DiffGDM_ens` combines structural guides. The evaluation API supports `decycle` and `direct` postprocessing. Hyperparameter sensitivity is handled by `dismantle_model_params_analysis.py` and evaluates direct node removal followed by reinsertion.

## Repository structure

```text
model_train/                Model training, losses and configurations
model_test/                 Inference, baseline algorithms and evaluation
  analysis_visi/            Original plotting and graph-analysis functions
  toy_model_analysis.py     Figure 5 Toy-network experiment
experiments_figure/         Manuscript figure entry points and local source data
```

## Citation

If you use this implementation, please cite the associated manuscript:

```bibtex
@misc{zhang2026bridging,
  title = {Bridging the Decycling-Dismantling Gap: A Structure-Guided Approach for Network Dismantling},
  author = {Zhang, Peng and Fu, Jun and Findeisen, Rolf},
  year = {2026},
  note = {Manuscript}
}
```

## License

This project is released under the [MIT License](LICENSE). Third-party baseline implementations retain their respective licenses.
