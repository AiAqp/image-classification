# Image-Classification
A PyTorch-based project for image classification experiments, featuring rapid model prototyping through dynamic configuration.

## Installation

### Using Conda:
Create and activate the environment:
```bash
conda env create -f conda.yaml
conda activate image-classification
```

### Using Pip:
Ensure a virtual environment (Python 3.9) is activated:
```bash
pip install -r requirements.txt
```

## Usage

### Configuration
This project uses Hydra for configuration managment. Modify main configuration file `conf/config.yaml` for custom experiment settings.

### Running Experiments
Execute the main script with optional configuration overrides:
```bash
python main.py training.n_epochs=5 experiment.use_gpu=False

```

### Experiment Tracking with MLflow
Launch MLflow's UI for tracking experiments. Use optional `--port` for a custom port (default access at `http://localhost:5000`):
```bash
mlflow ui --port 8080
```