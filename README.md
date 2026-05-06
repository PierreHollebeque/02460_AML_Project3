# Mini-project 3 in Advanced Machine Learning (02460)

*Advanced Machine Learning : 02460-2026 Mini-project 3 s039894, s271828 & s314159*  
*Mini-project 3 in Advanced Machine Learning (02460) by Mads ALKJAERSIG (s203372), Rémi BERTHELOT (s254144) & Pierre HOLLEBEQUE (s254136) on 07 May 2026*

This project implements a Discrete Denoising Diffusion Probabilistic Model (DDPM) for graph generation. The diffusion process iteratively adds categorical noise to graph features (nodes, edges) and learns a reverse process using a Graph Transformer neural network to generate novel graphs.

## Project Architecture

The codebase is modularized into several components:

- **`main.py`**: The main entry point. Provides a Command Line Interface (CLI) to train models, generate graph samples, perform hyperparameter searches, and evaluate generated graphs against baselines.
- **`ddpm.py`**: Contains the core logic for the Discrete DDPM, including the transition matrices ($Q$), the forward diffusion process, the reverse sampling process, and the cross-entropy loss computation.
- **`network.py` & `layers.py`**: Implements the `GraphTransformer` network used to predict the denoised discrete distributions of nodes and edges at each reverse diffusion step. It leverages multi-head attention and feature integration blocks (NodeEdgeBlock).
- **`train.py`**: Contains the training loop and learning rate scheduler integration, updating the model parameters and optionally plotting the training loss.
- **`baseline.py`**: Implements the Erdös-Rényi (ER) baseline graph generation logic by extracting edge densities from the training dataset.
- **`evaluate.py` & `graph_stat.py`**: Evaluation tools to compare generated graphs with baseline and training sets. Evaluates properties such as novelty, uniqueness (via Weisfeiler-Lehman hashing), node degrees, clustering coefficients, and eigenvector centralities.
- **`utils.py` & `diffusion_utils.py`**: Helper scripts for tasks including loading datasets, formatting structures (`PlaceHolder`, `DatasetInfos`), sampling discrete feature noise, computing posteriors distributions, and scheduling noise.
- **`dataset_distribution_plot.py`**: Diagnostic utility for plotting the empirical distributions of nodes and edges in the MUTAG dataset.

## Usage (`main.py`)

The `main.py` script requires a positional argument `mode` to specify the operation, followed by several optional flags to control model behavior and hyperparameters.

### Basic Command Structure
```bash
python main.py <mode> [OPTIONS]
```

### Modes

#### 1. Train the Model (`train`)
Trains the DDPM model on the dataset. Saves the resulting model to `--model-path`.
```bash
python main.py train --epochs 50 --batch-size 32 --lr 1e-3 --model-path model.pt
```
- `--num-hidden`: Number of hidden units (default: 128).
- `--n-layers`: Number of GraphTransformer layers (default: 4).
- `--T`: Number of diffusion steps (default: 100).

#### 2. Sample from the Model (`sample`)
Generates graphs using a pre-trained model and prints out the generated adjacency matrices.
```bash
python main.py sample --model-path model.pt --num-sample 5 --batch-size 5
```

#### 3. Hyperparameter Search (`hyperparameter_search`)
Performs a grid search over hyperparameters provided in a JSON file, logs the best loss, and plots the loss curves.
```bash
python main.py hyperparameter_search --hparams-search-file params.json
```
*Note: The `params.json` file should contain arrays for keys like `T`, `num_hidden`, and `n_layers`.*

#### 4. Baseline Generation (`baseline`)
Generates synthetic graphs strictly from the structured baseline distribution (Erdös-Rényi) and prints the matrices.
```bash
python main.py baseline --num-sample 5
```

#### 5. Compare & Compute Statistics (`stats`)
Evaluates the model by generating graphs, generating baseline ER graphs, and evaluating them against the true training dataset for Uniqueness and Novelty.
```bash
python main.py stats --model-path model.pt --num-sample 100
```

### Common Options
- `--model-path`: File to save the model to, or load from. (Default: `model.pt`)
- `--device`: Execution device (`cpu`, `cuda`, `mps`). (Default: `cpu`)
- `--batch-size`: Number of samples per batch during training/sampling. (Default: 32)
- `--num-sample`: Amount of generated graph output requested. (Default: 1)