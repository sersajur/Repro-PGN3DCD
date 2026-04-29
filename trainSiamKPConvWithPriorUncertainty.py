import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer_SiamKPConv import Trainer


@hydra.main(config_path="conf", config_name="configSiamKPConvWithPriorUncertainty")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    trainer = Trainer(cfg)
    trainer.train()
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
