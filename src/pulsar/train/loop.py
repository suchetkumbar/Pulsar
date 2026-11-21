"""Plain-PyTorch training / prediction loop for TraceNet."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pulsar.aug.pipeline import AugConfig
from pulsar.data.windows import Windows
from pulsar.eval.metrics import fairness_summary, ita_group, subject_level
from pulsar.eval.selective import fit_temperature
from pulsar.models.losses import total_loss
from pulsar.models.tracenet import TraceNet, TraceNetConfig, hr_from_logits
from pulsar.train.dataset import WindowDataset
from pulsar.train.runlog import RunLog


@dataclass
class TrainConfig:
    epochs: int = 30
    batch: int = 64
    lr: float = 1e-3
    wd: float = 1e-2
    warmup_epochs: int = 2
    workers: int = 0
    seed: int = 0
    device: str = "auto"
    w_pearson: float = 1.0
    w_ce: float = 1.0
    w_rel: float = 0.1
    supervise_rel: bool = True
    group_col: str = "ita_group"  # validation worst-group selection
    select: str = "worst_group_mae"  # or "mae"
    grad_clip: float = 5.0


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _to(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict(
    model: TraceNet,
    ds: WindowDataset,
    device,
    batch: int = 128,
    temperature: float | None = None,
    return_logits: bool = False,
):
    """Per-window predictions. temperature=None uses the model's calibrated temperature."""
    model.eval()
    temperature = float(getattr(model, "temperature", 1.0)) if temperature is None else temperature
    rows, logits_all = [], []
    for b in DataLoader(ds, batch_size=batch, shuffle=False):
        b = _to(b, device)
        out = model(b["x"], b["valid"], b["n_pix"], b["motion"])
        if return_logits:
            logits_all.append(out["hr_logits"].float().cpu().numpy())
        hr_soft, hr_mode, conf = hr_from_logits(out["hr_logits"], model.bpm, temperature)
        any_valid = out["patch_mask"].any(1)
        for i, k in enumerate(b["key"].tolist()):
            m = ds.window_meta(k)
            rows.append(
                m
                | {
                    "hr_true": float(b["hr"][i]),
                    "hr_pred": float(hr_soft[i]),
                    "hr_mode": float(hr_mode[i]),
                    "conf": float(conf[i]),
                    "any_valid": bool(any_valid[i]),
                    "ita_group": ita_group(m.get("ita")),
                }
            )
    df = pd.DataFrame(rows)
    if return_logits:
        return df, (np.concatenate(logits_all) if logits_all else np.zeros((0, model.bpm.numel())))
    return df


def _checkpoint(model: TraceNet, model_cfg: TraceNetConfig, region_ids, temperature: float = 1.0) -> dict:
    return {
        "model": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "region_ids": np.asarray(region_ids).tolist(),
        "temperature": float(temperature),
    }


def evaluate_df(df: pd.DataFrame, group_col: str) -> dict:
    out = {"mae": float(subject_level(df)["mae"].iloc[0])}
    if group_col in df and df[group_col].nunique() > 1:
        fs = fairness_summary(df, group_col)
        out |= {"worst_group_mae": fs["worst_group_mae"], "gap": fs["gap"], "worst_group": fs["worst_group"]}
    else:
        out |= {"worst_group_mae": out["mae"], "gap": 0.0, "worst_group": "all"}
    return out


def train(
    train_ws: list[Windows],
    val_ws: list[Windows],
    model_cfg: TraceNetConfig,
    cfg: TrainConfig,
    aug: AugConfig | None,
    region_ids: np.ndarray,
    run_name: str = "tracenet",
    config_dump: dict | None = None,
    runs_root: str | Path = "runs",
) -> tuple[TraceNet, RunLog, dict]:
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model = TraceNet(model_cfg, region_ids).to(device)
    log = RunLog(
        run_name,
        (config_dump or {})
        | {
            "model": asdict(model_cfg),
            "train": asdict(cfg),
            "aug": repr(aug),
            "n_params": model.num_params(),
        },
        root=runs_root,
    )
    ds_tr = WindowDataset(train_ws, aug, model_cfg.fs, cfg.seed)
    ds_va = WindowDataset(val_ws, None, model_cfg.fs)
    dl = DataLoader(
        ds_tr,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.workers,
        drop_last=True,
        persistent_workers=cfg.workers > 0,
        pin_memory=device.type == "cuda",
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    steps = max(1, cfg.epochs * len(dl))
    warm = cfg.warmup_epochs * len(dl)

    def lr_at(step: int) -> float:
        if step < warm:
            return (step + 1) / max(1, warm)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    best, best_metrics = float("inf"), {}
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        ds_tr.set_epoch(epoch)
        t0, agg, n = time.time(), {}, 0
        for b in dl:
            b = _to(b, device)
            out = model(b["x"], b["valid"], b["n_pix"], b["motion"])
            loss, logs = total_loss(
                out,
                b,
                model.bpm,
                model_cfg.fs,
                cfg.w_pearson,
                cfg.w_ce,
                cfg.w_rel,
                cfg.supervise_rel and model_cfg.adaptive,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            n += 1
            for k, v in (logs | {"loss": float(loss)}).items():
                agg[k] = agg.get(k, 0.0) + v
        row = {"epoch": epoch, "sec": round(time.time() - t0, 1), "lr": sched.get_last_lr()[0]}
        row |= {k: v / max(1, n) for k, v in agg.items()}
        if len(ds_va):
            ev = evaluate_df(predict(model, ds_va, device), cfg.group_col)
            row |= {f"val_{k}": v for k, v in ev.items()}
            score = ev[cfg.select]
            if score < best:
                best, best_metrics = score, ev | {"epoch": epoch}
                torch.save(_checkpoint(model, model_cfg, region_ids), log.dir / "best.pt")
        log.log(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    torch.save(_checkpoint(model, model_cfg, region_ids), log.dir / "last.pt")
    if (log.dir / "best.pt").exists():
        model.load_state_dict(torch.load(log.dir / "best.pt", map_location=device)["model"])
    # C4: post-hoc temperature calibration on validation (docs 04 §7)
    temperature = 1.0
    if len(ds_va):
        val_df, val_logits = predict(model, ds_va, device, temperature=1.0, return_logits=True)
        temperature = fit_temperature(val_logits, val_df["hr_true"].to_numpy(), model.bpm.cpu().numpy())
        torch.save(_checkpoint(model, model_cfg, region_ids, temperature), log.dir / "best.pt")
    model.temperature = temperature
    best_metrics["temperature"] = temperature
    log.summary({"best": best_metrics, "n_train_windows": len(ds_tr), "n_val_windows": len(ds_va)})
    return model, log, best_metrics


def load_model(path: str | Path, device="cpu") -> TraceNet:
    ck = torch.load(path, map_location=device)
    model = TraceNet(TraceNetConfig(**ck["model_cfg"]), np.asarray(ck["region_ids"]))
    model.load_state_dict(ck["model"])
    model.temperature = float(ck.get("temperature", 1.0))
    return model.to(device).eval()
