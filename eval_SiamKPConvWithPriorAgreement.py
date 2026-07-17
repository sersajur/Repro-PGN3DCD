import hydra
import logging
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer_SiamKPConv import Trainer
from torch_points3d.utils.repro import set_deterministic, collect_repro_stamp, log_repro_stamp

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="evalSiamKPConvWithPriorAgreement")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    seed = cfg.get("seed", 42)
    set_deterministic(seed=seed)
    log_repro_stamp(collect_repro_stamp(seed=seed))
    if cfg.pretty_print:
        log.info(f"Start evaluation with config:\n{OmegaConf.to_yaml(cfg)}")

    trainer = Trainer(cfg)
    trainer.eval()
    # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
