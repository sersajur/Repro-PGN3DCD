import hydra
import logging
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer_SiamKPConv import Trainer

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="evalSiamKPConv")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        log.info(f"Start evaluation with config:\n{OmegaConf.to_yaml(cfg)}")

    trainer = Trainer(cfg)
    trainer.eval()
    # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
