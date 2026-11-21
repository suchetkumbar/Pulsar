"""Train TraceNet (Hydra entry point). See configs/train.yaml for options."""

from __future__ import annotations

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from pulsar.aug.illumsim import IllumSimConfig
from pulsar.aug.phystone import PhysToneConfig
from pulsar.aug.pipeline import AugConfig
from pulsar.data.loading import load_cache_windows, split_subjects, synthetic_windows
from pulsar.models.tracenet import TraceNetConfig
from pulsar.preprocess.patches import PatchSpec
from pulsar.train.loop import TrainConfig, train


def build_aug(c: DictConfig) -> AugConfig:
    return AugConfig(
        standard=c.standard,
        naive_tone_p=c.naive_tone_p,
        phystone=PhysToneConfig(
            p_apply=c.phystone_p, surface_frac=c.surface_frac, noise_topup=c.phystone_noise_topup
        )
        if c.phystone
        else None,
        illumsim=IllumSimConfig() if c.illumsim else None,
        illumsim_p=c.illumsim_p,
    )


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    d = cfg.data
    if d.source == "synthetic":
        s = d.synthetic
        ws = synthetic_windows(
            s.n_subjects, s.videos_per_subject, s.duration_s, d.fs, d.win_s, s.stride_s, seed=cfg.train.seed
        )
    else:
        ws = load_cache_windows(list(d.caches), d.fs, d.win_s, d.stride_s)
    tr, va, _ = split_subjects(ws, d.val_frac, d.test_frac, seed=0, split_file=d.split_file)
    print(
        f"windows: train={sum(map(len, tr))} ({len(tr)} videos), val={sum(map(len, va))} ({len(va)} videos)"
    )
    model_cfg = TraceNetConfig(
        width=cfg.model.width,
        enc_blocks=cfg.model.enc_blocks,
        dec_blocks=cfg.model.dec_blocks,
        adaptive=cfg.model.adaptive,
        fs=d.fs,
        t_len=int(round(d.win_s * d.fs)),
    )
    tcfg = TrainConfig(**{k: v for k, v in OmegaConf.to_container(cfg.train).items()})
    region_ids = PatchSpec().region_ids
    _, log, best = train(
        tr,
        va,
        model_cfg,
        tcfg,
        build_aug(cfg.aug),
        np.asarray(region_ids),
        cfg.name,
        OmegaConf.to_container(cfg, resolve=True),
        cfg.runs_root,
    )
    print(f"run dir: {log.dir}\nbest: {best}")


if __name__ == "__main__":
    main()
