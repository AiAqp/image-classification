import hydra
from omegaconf import DictConfig

from experiment import ExperimentRunner

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    root = hydra.utils.get_original_cwd()

    experiment = ExperimentRunner(cfg)
    experiment.run()

if __name__ == '__main__':
    main()