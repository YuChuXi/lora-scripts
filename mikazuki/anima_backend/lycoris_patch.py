from __future__ import annotations

import importlib
import importlib.metadata
import logging

import torch


log = logging.getLogger(__name__)


def _lycoris_lora_version() -> str | None:
    try:
        return importlib.metadata.version("lycoris-lora")
    except importlib.metadata.PackageNotFoundError:
        return None


def _cast_delta_weight(delta_weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Align a LyCORIS delta-weight with the module input dtype and device.

    DoRA weight decomposition promotes the weight to float32 internally, so
    the resulting delta would otherwise enter ``self.op(x, delta_weight, ..)``
    with a float dtype while the input stays bfloat16 and fail on ``F.linear``.

    Args:
        delta_weight (torch.Tensor): Weight delta produced by the module.
        x (torch.Tensor): Module input tensor.

    Returns:
        torch.Tensor: ``delta_weight`` cast to the input dtype and device.

    Callers:
        - mikazuki.anima_backend.lycoris_patch._patched_lokr_forward
        - mikazuki.anima_backend.lycoris_patch._patched_locon_forward
    """
    if x.is_floating_point() and delta_weight.dtype != x.dtype:
        return delta_weight.to(device=x.device, dtype=x.dtype)
    return delta_weight.to(device=x.device)


def _patched_lokr_forward(self, x: torch.Tensor, *args, **kwargs):
    """Run the stock LoKr forward with delta-weight dtype alignment.

    Line-for-line mirror of ``LokrModule.forward`` (LyCORIS 3.3) plus the
    shared delta-weight cast before ``self.op``.

    Args:
        self (lycoris.modules.lokr.LokrModule): Bound LyCORIS LoKr module.
        x (torch.Tensor): Module input tensor.

    Returns:
        torch.Tensor: Module output with the LyCORIS delta applied.

    Callers:
        - Assigned to ``lycoris.modules.lokr.LokrModule.forward`` by
          :func:`patch_lokr_dora_bf16_forward`.
    """
    if self.module_dropout and self.training:
        if torch.rand(1, device=x.device) < self.module_dropout:
            return self.org_forward(x, *args, **kwargs)

    if self.bypass_mode:
        return self.bypass_forward(x, self.multiplier)

    base = self.org_forward(x, *args, **kwargs)
    base_weight = self._current_weight().to(x.device)
    diff_weight = self.get_weight(self.shape).to(base_weight.dtype) * self.scalar

    if self.wd:
        new_weight = self.apply_weight_decompose(
            base_weight + diff_weight, self.multiplier
        )
    elif self.multiplier == 1:
        new_weight = base_weight + diff_weight
    else:
        new_weight = base_weight + diff_weight * self.multiplier

    delta_weight = new_weight - base_weight
    delta_weight = _cast_delta_weight(delta_weight, x)
    delta = self.op(x, delta_weight, None, **self.kw_dict)
    return base + delta


def _patched_locon_forward(self, x: torch.Tensor, *args, **kwargs):
    """Run the stock LoCon forward with delta-weight dtype alignment.

    Line-for-line mirror of ``LoConModule.forward`` (LyCORIS 3.3) plus the
    shared delta-weight cast before ``self.op``; same root cause as the
    LoKr patch (#161) on the LoCon branch used by ``algo=lora`` + DoRA.

    Args:
        self (lycoris.modules.locon.LoConModule): Bound LyCORIS LoCon module.
        x (torch.Tensor): Module input tensor.

    Returns:
        torch.Tensor: Module output with the LyCORIS delta applied.

    Callers:
        - Assigned to ``lycoris.modules.locon.LoConModule.forward`` by
          :func:`patch_locon_dora_bf16_forward`.
    """
    if self.module_dropout and self.training:
        if torch.rand(1) < self.module_dropout:
            return self.org_forward(x, *args, **kwargs)

    if self.bypass_mode:
        return self.bypass_forward(x, scale=self.multiplier)

    base = self.org_forward(x, *args, **kwargs)
    scale = self.scale
    device = x.device

    base_weight = self._current_weight().to(device)
    diff_weight = self.make_weight(device).to(base_weight.dtype) * scale
    if self.wd:
        new_weight = self.apply_weight_decompose(
            base_weight + diff_weight, self.multiplier
        )
    else:
        new_weight = base_weight + diff_weight * self.multiplier

    delta_weight = new_weight - base_weight
    delta_weight = _cast_delta_weight(delta_weight, x)
    delta = self.op(x, delta_weight, None, **self.kw_dict)
    return base + delta


def patch_lokr_dora_bf16_forward() -> bool:
    """Patch LyCORIS 3.3 LoKr DoRA delta-weight dtype mismatch (#161).

    Args:
        None

    Returns:
        bool: True when the patch was applied, False otherwise.

    Callers:
        - mikazuki.anima_backend.lycoris_patch.patch_lycoris_dora_bf16_forward
        - tests/test_anima_lycoris_patch.py:AnimaLycorisPatchTests
    """
    version = _lycoris_lora_version()
    if version != "3.3.0":
        return False

    try:
        lokr = importlib.import_module("lycoris.modules.lokr")
    except Exception as exc:
        log.warning("Could not import LyCORIS LoKr for dtype patch: %s", exc)
        return False

    module_cls = getattr(lokr, "LokrModule", None)
    if module_cls is None or getattr(module_cls, "_sd_trainer_lokr_dora_bf16_patch", False):
        return False

    module_cls.forward = _patched_lokr_forward
    module_cls._sd_trainer_lokr_dora_bf16_patch = True
    log.info("Applied LyCORIS 3.3 LoKr DoRA bf16 dtype patch.")
    return True


def patch_locon_dora_bf16_forward() -> bool:
    """Patch LyCORIS 3.3 LoCon DoRA delta-weight dtype mismatch.

    Same root cause as the LoKr patch (#161): ``apply_weight_decompose``
    promotes weights to float32 while the module input stays bfloat16,
    which crashes ``lycoris.modules.locon.LoConModule.forward`` used by
    ``algo=lora`` networks with DoRA enabled.

    Args:
        None

    Returns:
        bool: True when the patch was applied, False otherwise.

    Callers:
        - mikazuki.anima_backend.lycoris_patch.patch_lycoris_dora_bf16_forward
        - tests/test_anima_lycoris_patch.py:AnimaLycorisPatchTests
    """
    version = _lycoris_lora_version()
    if version != "3.3.0":
        return False

    try:
        locon = importlib.import_module("lycoris.modules.locon")
    except Exception as exc:
        log.warning("Could not import LyCORIS LoCon for dtype patch: %s", exc)
        return False

    module_cls = getattr(locon, "LoConModule", None)
    if module_cls is None or getattr(module_cls, "_sd_trainer_locon_dora_bf16_patch", False):
        return False

    module_cls.forward = _patched_locon_forward
    module_cls._sd_trainer_locon_dora_bf16_patch = True
    log.info("Applied LyCORIS 3.3 LoCon DoRA bf16 dtype patch.")
    return True


def patch_lycoris_dora_bf16_forward() -> None:
    """Apply the DoRA bf16 dtype patch to every supported LyCORIS module.

    Args:
        None

    Returns:
        None

    Callers:
        - scripts/dev/anima_train_network.py:main
    """
    patch_lokr_dora_bf16_forward()
    patch_locon_dora_bf16_forward()
