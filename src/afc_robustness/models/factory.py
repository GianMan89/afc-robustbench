"""Model factory for configuration-driven experiments."""

from __future__ import annotations

from typing import Any

from afc_robustness.models.acm_svm import ACMSVM
from afc_robustness.models.casim import CASIM
from afc_robustness.models.eac import EAC1NN
from afc_robustness.models.jaccard import Jaccard1NN
from afc_robustness.models.mbw_lr import MBWLogisticRegression
from afc_robustness.models.prefix import PrefixTrainedAFCModel
from afc_robustness.models.wdi import WDI1NN

MODEL_REGISTRY = {
    "wdi_1nn": WDI1NN,
    "WDI-1NN": WDI1NN,
    "jac_1nn": Jaccard1NN,
    "JAC-1NN": Jaccard1NN,
    "eac_1nn": EAC1NN,
    "EAC-1NN": EAC1NN,
    "mbw_lr": MBWLogisticRegression,
    "MBW-LR": MBWLogisticRegression,
    "acm_svm": ACMSVM,
    "ACM-SVM": ACMSVM,
    "casim": CASIM,
    "CASIM": CASIM,
}

_PREFIX_KEYS = {
    "prefix_grid",
    "prefix_reference",
    "prefix_train_reference",
    "prefix_selection",
    "prefix_train_horizon",
    "prefix_min_time_steps",
    "include_full_prefix",
}

_PREFIX_MODES = {"prefix", "prefix_ensemble", "stagewise", "stepwise"}


def _pop_prefix_wrapper_params(params: dict[str, Any], *, default_min_time_steps: int) -> dict[str, Any]:
    wrapper_params: dict[str, Any] = {}
    for key in list(params):
        if key in _PREFIX_KEYS:
            wrapper_params[key] = params.pop(key)
    wrapper_params.setdefault("prefix_min_time_steps", default_min_time_steps)
    return wrapper_params


def make_model(name: str, params: dict[str, Any] | None = None):
    """Instantiate a model by registry name.

    A model can be wrapped in the generic prefix-trained meta-estimator by
    passing ``training_mode: prefix`` in the model's parameter dictionary. All
    normal hyperparameters stay with the base classifier; ``prefix_*`` keys
    configure the wrapper.
    """

    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model name: {name}")

    model_cls = MODEL_REGISTRY[name]
    params = {} if params is None else dict(params)
    training_mode = str(params.pop("training_mode", params.pop("training_strategy", "full"))).lower()

    if training_mode in _PREFIX_MODES:
        default_min_time_steps = 9 if model_cls is CASIM else 1
        wrapper_params = _pop_prefix_wrapper_params(
            params,
            default_min_time_steps=default_min_time_steps,
        )
        return PrefixTrainedAFCModel(
            base_model_cls=model_cls,
            base_params=params,
            **wrapper_params,
        )

    if training_mode not in {"full", "default", "none"}:
        raise ValueError(
            "training_mode must be 'full' or one of "
            f"{sorted(_PREFIX_MODES)}; got {training_mode!r}"
        )

    return model_cls(**params)


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)
