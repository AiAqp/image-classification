import hydra
from omegaconf import DictConfig, OmegaConf
from utils import load_config
from experiment import ExperimentTracker

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    root = hydra.utils.get_original_cwd()

    experiment = ExperimentTracker(cfg)
    experiment.run()

if __name__ == '__main__':
    main()