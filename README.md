<p align="center">
  <img src="retina/assets/logo.png" alt="Retina Logo" width="400"/>
</p>

## Overview

Retinal fundus classifier we made for a school project

## Installation

```bash
# Clone the repository
git clone https://github.com/ruyoga/retina.git
cd retina

# Add data to retina/retina/data

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

To initiate model training:

```bash
python -m retina.experiments.train
```

### Hyperparameter Tuning

To perform hyperparameter tuning with Weights & Biases:

1. Initialize the sweep:
```bash
python -m retina.experiments.init_sweep
```

2. Run the sweep agent:
```bash
wandb agent
```
