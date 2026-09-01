"""Tiny logger that fans out to stdout + TensorBoard + (optional) WandB.

Ported from merits-l-llama. A one-stage run is a single multi-hour job whose
number lands in a thesis table, so WandB is on by default (see the
`wandb-for-long-training` rule).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:  # pragma: no cover
    _WANDB_AVAILABLE = False


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class RunLogger:
    def __init__(
        self,
        output_dir: str | Path,
        run_name: str,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.tb = SummaryWriter(log_dir=str(self.output_dir / "tb"))

        self.logger = logging.getLogger(run_name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                  datefmt="%H:%M:%S")
            )
            self.logger.addHandler(handler)
            file_handler = logging.FileHandler(self.output_dir / "train.log", encoding="utf-8")
            file_handler.setFormatter(handler.formatter)
            self.logger.addHandler(file_handler)

        self.metrics_path = self.output_dir / "metrics.jsonl"

        self.use_wandb = bool(use_wandb and _WANDB_AVAILABLE)
        if use_wandb and not _WANDB_AVAILABLE:
            self.logger.warning("use_wandb=True but `wandb` not installed — falling back to TB only.")
        if self.use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    name=run_name,
                    dir=str(self.output_dir),
                    config=_to_plain(wandb_config or {}),
                    reinit=True,
                )
                self.logger.info(f"WandB run: {wandb.run.url}")
            except Exception as e:  # noqa: BLE001
                self.logger.warning(
                    f"wandb.init failed ({e}); falling back to TB only. "
                    f"Hint: run `wandb login` or set WANDB_API_KEY."
                )
                self.use_wandb = False

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def update_wandb_config(self, updates: Dict[str, Any]) -> None:
        """Merge additional key/values into the WandB run config (safe if wandb disabled)."""
        if self.use_wandb:
            try:
                wandb.config.update(_to_plain(updates), allow_val_change=True)
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"wandb.config.update failed: {e}")

    def log_scalars(self, scalars: Dict[str, float], step: int, prefix: str = "") -> None:
        for k, v in scalars.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.tb.add_scalar(tag, v, step)
        if self.use_wandb:
            wandb.log({f"{prefix}/{k}" if prefix else k: v for k, v in scalars.items()}, step=step)
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "prefix": prefix, **scalars}) + "\n")

    def close(self) -> None:
        self.tb.flush()
        self.tb.close()
        if self.use_wandb:
            wandb.finish()
