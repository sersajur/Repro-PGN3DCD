import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from torch_points3d.trainer_SiamKPConv import Trainer
from torch_points3d.utils.repro import set_deterministic, collect_repro_stamp, log_repro_stamp


@hydra.main(config_path="conf", config_name="configSiamKPConvWithPriorAgreement")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    seed = cfg.get("seed", 42)
    set_deterministic(seed=seed)
    stamp = collect_repro_stamp(seed=seed)
    log_repro_stamp(stamp)
    cfg._repro_stamp = stamp

    trainer = Trainer(cfg)
    trainer.train()
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
