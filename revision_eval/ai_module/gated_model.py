"""CPU-only building blocks for physics-guided gated event reliability.

The physics branch in this module consumes a *VGS-derived continuous score*.
It is a calibrated auxiliary reliability signal and must not be described as,
or substituted for, the complete historical VGS pipeline reported in the
original manuscript.  This file deliberately contains no data loading,
splitting, corruption, or downstream mechanics evaluation code.

The intended staged use is:

1. fit ``AIExpert`` on training data;
2. fit ``PhysicsCalibrator`` with *training-split* proxy labels only;
3. freeze both experts and train ``ReliabilityGate`` with ``gated_loss``;
4. choose an event-selection threshold on validation data only.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ArrayLike = Union[np.ndarray, Tensor, Sequence[float]]
_EPS = 1.0e-8


class InputValidationError(ValueError):
    """Raised when an event-reliability tensor violates the frozen interface."""


def set_cpu_seed(seed: int) -> None:
    """Set CPU random seeds for deterministic small-model initialization/training.

    The module is intentionally CPU-only.  Call this before a training run when
    reproducible initialization and minibatch order are required.
    """
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _as_cpu_float(value: ArrayLike, name: str) -> Tensor:
    """Convert an input to a CPU float tensor without silently accepting CUDA."""
    if isinstance(value, Tensor):
        if value.device.type != "cpu":
            raise InputValidationError(f"{name} must be a CPU tensor")
        tensor = value.to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if not torch.isfinite(tensor).all().item():
        raise InputValidationError(f"{name} contains NaN or infinity")
    return tensor


def _as_column(value: ArrayLike, name: str) -> Tensor:
    tensor = _as_cpu_float(value, name)
    if tensor.ndim == 1:
        tensor = tensor[:, None]
    if tensor.ndim != 2 or tensor.shape[1] != 1:
        raise InputValidationError(f"{name} must have shape (N,) or (N, 1), got {tuple(tensor.shape)}")
    return tensor


def _as_features(value: ArrayLike, name: str, width: int) -> Tensor:
    tensor = _as_cpu_float(value, name)
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise InputValidationError(f"{name} must have shape (N, {width}), got {tuple(tensor.shape)}")
    return tensor


def _same_rows(*pairs: Tuple[str, Tensor]) -> int:
    if not pairs:
        raise RuntimeError("at least one tensor is required")
    count = pairs[0][1].shape[0]
    for name, value in pairs[1:]:
        if value.shape[0] != count:
            raise InputValidationError(
                f"{name} has {value.shape[0]} rows, expected {count} to match {pairs[0][0]}"
            )
    return count


def _check_probability(value: ArrayLike, name: str, *, eps: float = _EPS) -> Tensor:
    tensor = _as_column(value, name)
    # Exact 0 and 1 are permitted at the interface.  BCE operations below use
    # a numerically safe clamp internally.
    if (tensor.detach() < 0).any().item() or (tensor.detach() > 1).any().item():
        raise InputValidationError(f"{name} must lie in [0, 1]")
    return tensor


def _check_labels(labels: ArrayLike, name: str = "labels") -> Tensor:
    tensor = _as_column(labels, name)
    if (tensor.detach() < 0).any().item() or (tensor.detach() > 1).any().item():
        raise InputValidationError(f"{name} must lie in [0, 1]")
    # Proxy labels are expected to be binary rather than soft reliability
    # targets.  A tiny tolerance only protects conversions from numpy float.
    if not torch.all((tensor.detach() <= 1e-6) | (tensor.detach() >= 1 - 1e-6)).item():
        raise InputValidationError(f"{name} must contain binary 0/1 proxy labels")
    return tensor


def _safe_bce(probability: Tensor, labels: Tensor, eps: float = _EPS) -> Tensor:
    # ``1 - 1e-8`` rounds back to one in float32.  Use the dtype epsilon as a
    # minimum margin so exact sigmoid endpoint inputs remain finite in BCE.
    margin = max(float(eps), float(torch.finfo(probability.dtype).eps))
    probability = probability.clamp(min=margin, max=1.0 - margin)
    return F.binary_cross_entropy(probability, labels, reduction="none")


def _assert_cpu_module(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        if parameter.device.type != "cpu":
            raise RuntimeError(f"{module.__class__.__name__}.{name} is not on CPU")


class PhysicsCalibrator(nn.Module):
    """Two-parameter Platt calibration of a non-negative VGS-derived score.

    ``score`` can be zero: it is transformed as ``log(score + eps)``.  Scores
    are continuous physics evidence, not probabilities, and no upper bound is
    imposed.  The calibration parameters must be fitted only on the training
    split using :func:`fit_platt`, then frozen before validation/test inference.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0, eps: float = _EPS) -> None:
        super().__init__()
        if not math.isfinite(float(a)) or not math.isfinite(float(b)):
            raise ValueError("Platt parameters must be finite")
        if not math.isfinite(float(eps)) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.a = nn.Parameter(torch.tensor(float(a), dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(float(b), dtype=torch.float32))
        self.eps = float(eps)
        self.fit_metadata: Dict[str, Union[int, float, bool, str]] = {}

    def score_logit(self, scores: ArrayLike) -> Tensor:
        """Return ``log(score + eps)`` after non-negative/finite checks."""
        scores_t = _as_column(scores, "physics_scores")
        if (scores_t.detach() < 0).any().item():
            raise InputValidationError("physics_scores must be non-negative; zero is valid")
        return torch.log(scores_t + self.eps)

    def forward(self, scores: ArrayLike, *, return_logits: bool = False) -> Tensor:
        _assert_cpu_module(self)
        logits = self.a * self.score_logit(scores) + self.b
        return logits if return_logits else torch.sigmoid(logits)

    def freeze(self) -> "PhysicsCalibrator":
        self.requires_grad_(False)
        return self

    def unfreeze(self) -> "PhysicsCalibrator":
        self.requires_grad_(True)
        return self

    def calibration_dict(self) -> Dict[str, Union[int, float, bool, str]]:
        """Return serializable parameters and training provenance reminder."""
        return {
            "a": float(self.a.detach()),
            "b": float(self.b.detach()),
            "eps": self.eps,
            "score_transform": "log(score + eps)",
            "score_semantics": "VGS-derived continuous score; not historical full VGS",
            **self.fit_metadata,
        }


def fit_platt(
    train_scores: ArrayLike,
    train_labels: ArrayLike,
    *,
    class_balanced: bool = True,
    max_iter: int = 200,
    lr: float = 0.5,
    eps: float = _EPS,
    seed: Optional[int] = None,
) -> PhysicsCalibrator:
    """Fit a Platt calibrator using *only caller-supplied training-split data*.

    This function has no split-aware data access by design.  Its explicit
    ``train_*`` arguments are a protocol guard: callers must never pass
    validation, held-out, EBIV, or EBOS samples to it.  When ``class_balanced``
    is true, source and injected proxy-label classes have equal aggregate loss
    weight regardless of their event count.
    """
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not math.isfinite(float(lr)) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if seed is not None:
        set_cpu_seed(seed)

    scores = _as_column(train_scores, "train_scores")
    labels = _check_labels(train_labels, "train_labels")
    _same_rows(("train_scores", scores), ("train_labels", labels))
    if scores.shape[0] < 2:
        raise InputValidationError("Platt fitting needs at least two training samples")
    if (scores.detach() < 0).any().item():
        raise InputValidationError("train_scores must be non-negative; zero is valid")
    n_positive = int((labels > 0.5).sum().item())
    n_negative = int((labels <= 0.5).sum().item())
    if n_positive == 0 or n_negative == 0:
        raise InputValidationError("Platt fitting needs both binary proxy-label classes")

    prior = min(max(float(labels.mean()), eps), 1.0 - eps)
    calibrator = PhysicsCalibrator(a=1.0, b=math.log(prior / (1.0 - prior)), eps=eps)
    if class_balanced:
        n_total = float(len(labels))
        weights = torch.where(
            labels > 0.5,
            torch.full_like(labels, n_total / (2.0 * n_positive)),
            torch.full_like(labels, n_total / (2.0 * n_negative)),
        )
    else:
        weights = torch.ones_like(labels)

    optimizer = torch.optim.LBFGS(
        calibrator.parameters(), lr=float(lr), max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        probabilities = calibrator(scores)
        loss = (_safe_bce(probabilities, labels, eps) * weights).mean()
        if not torch.isfinite(loss).item():
            raise FloatingPointError("non-finite Platt calibration loss")
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        final_loss = float(((_safe_bce(calibrator(scores), labels, eps) * weights).mean()).item())
    calibrator.fit_metadata = {
        "fitted_on": "caller-supplied training split only",
        "training_samples": int(scores.shape[0]),
        "training_positive": n_positive,
        "training_negative": n_negative,
        "class_balanced": bool(class_balanced),
        "optimization": "CPU L-BFGS",
        "training_loss": final_loss,
    }
    return calibrator


class AIExpert(nn.Module):
    """Lightweight data-driven event expert: six generic descriptors → 1 logit."""

    input_dim = 6

    def __init__(self, *, seed: Optional[int] = None) -> None:
        super().__init__()
        if seed is not None:
            set_cpu_seed(seed)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def logits(self, generic_features: ArrayLike) -> Tensor:
        _assert_cpu_module(self)
        return self.network(_as_features(generic_features, "generic_features", self.input_dim))

    def forward(self, generic_features: ArrayLike, *, return_logits: bool = False) -> Tensor:
        logits = self.logits(generic_features)
        return logits if return_logits else torch.sigmoid(logits)


class ReliabilityGate(nn.Module):
    """Learned reliability gate: 17 causal gate inputs → AI contribution alpha."""

    input_dim = 17

    def __init__(self, *, seed: Optional[int] = None) -> None:
        super().__init__()
        if seed is not None:
            set_cpu_seed(seed)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def logits(self, gate_features: ArrayLike) -> Tensor:
        _assert_cpu_module(self)
        return self.network(_as_features(gate_features, "gate_features", self.input_dim))

    def forward(self, gate_features: ArrayLike, *, return_logits: bool = False) -> Tensor:
        logits = self.logits(gate_features)
        return logits if return_logits else torch.sigmoid(logits)


def make_gate_features(
    physics_features: ArrayLike,
    generic_features: ArrayLike,
    p_physics: ArrayLike,
    p_ai: ArrayLike,
) -> Tensor:
    """Build [8 physics, 6 generic, Pp, Pa, |Pp-Pa|] gate inputs of width 17."""
    physics = _as_features(physics_features, "physics_features", 8)
    generic = _as_features(generic_features, "generic_features", 6)
    p_p = _check_probability(p_physics, "p_physics")
    p_a = _check_probability(p_ai, "p_ai")
    _same_rows(
        ("physics_features", physics),
        ("generic_features", generic),
        ("p_physics", p_p),
        ("p_ai", p_a),
    )
    gate_features = torch.cat((physics, generic, p_p, p_a, torch.abs(p_p - p_a)), dim=1)
    if gate_features.shape[1] != ReliabilityGate.input_dim:
        raise RuntimeError("internal gate feature width drifted from 17")
    return gate_features


@dataclass
class GatedOutput:
    """Named intermediate probabilities for training, diagnostics, and figures."""

    p_physics: Tensor
    p_ai: Tensor
    alpha: Tensor
    p_fused: Tensor
    gate_features: Tensor

    def detached_numpy(self) -> Dict[str, np.ndarray]:
        return {
            "p_physics": self.p_physics.detach().cpu().numpy().reshape(-1),
            "p_ai": self.p_ai.detach().cpu().numpy().reshape(-1),
            "alpha": self.alpha.detach().cpu().numpy().reshape(-1),
            "p_fused": self.p_fused.detach().cpu().numpy().reshape(-1),
        }


class GatedDualExpert(nn.Module):
    """Physics-score calibrator + generic AI expert + learned 17-D gate.

    Its physics input is a VGS-derived continuous score.  It is intentionally
    not an implementation of the historical full VGS denoiser and must be
    reported as a newly trained AI-assisted model in a separate result row.
    """

    def __init__(
        self,
        physics_calibrator: Optional[PhysicsCalibrator] = None,
        ai_expert: Optional[AIExpert] = None,
        gate: Optional[ReliabilityGate] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            set_cpu_seed(seed)
        self.physics_calibrator = physics_calibrator if physics_calibrator is not None else PhysicsCalibrator()
        self.ai_expert = ai_expert if ai_expert is not None else AIExpert()
        self.gate = gate if gate is not None else ReliabilityGate()
        self.cpu()

    @staticmethod
    def fuse(p_physics: ArrayLike, p_ai: ArrayLike, alpha: ArrayLike) -> Tensor:
        """Continuous convex fusion ``(1-alpha) Pp + alpha Pa`` with guards."""
        p_p = _check_probability(p_physics, "p_physics")
        p_a = _check_probability(p_ai, "p_ai")
        alpha_t = _check_probability(alpha, "alpha")
        _same_rows(("p_physics", p_p), ("p_ai", p_a), ("alpha", alpha_t))
        fused = (1.0 - alpha_t) * p_p + alpha_t * p_a
        # A convex combination of validated probabilities must remain bounded;
        # retain this assertion so silent API changes cannot invalidate losses.
        if (fused.detach() < 0).any().item() or (fused.detach() > 1).any().item():
            raise RuntimeError("fusion produced probability outside [0, 1]")
        return fused

    def components(
        self,
        generic_features: ArrayLike,
        physics_features: ArrayLike,
        physics_scores: ArrayLike,
    ) -> GatedOutput:
        """Compute both experts, the gate, and fused reliability on CPU."""
        _assert_cpu_module(self)
        generic = _as_features(generic_features, "generic_features", AIExpert.input_dim)
        physics = _as_features(physics_features, "physics_features", 8)
        scores = _as_column(physics_scores, "physics_scores")
        if (scores.detach() < 0).any().item():
            raise InputValidationError("physics_scores must be non-negative; zero is valid")
        _same_rows(
            ("generic_features", generic),
            ("physics_features", physics),
            ("physics_scores", scores),
        )
        p_physics = self.physics_calibrator(scores)
        p_ai = self.ai_expert(generic)
        gate_features = make_gate_features(physics, generic, p_physics, p_ai)
        alpha = self.gate(gate_features)
        fused = self.fuse(p_physics, p_ai, alpha)
        return GatedOutput(p_physics, p_ai, alpha, fused, gate_features)

    def forward(
        self,
        generic_features: ArrayLike,
        physics_features: ArrayLike,
        physics_scores: ArrayLike,
        *,
        return_components: bool = False,
    ) -> Union[Tensor, GatedOutput]:
        output = self.components(generic_features, physics_features, physics_scores)
        return output if return_components else output.p_fused

    def freeze_experts(self) -> "GatedDualExpert":
        """Freeze calibration and AI weights for Stage-C gate-only training."""
        self.physics_calibrator.requires_grad_(False)
        self.ai_expert.requires_grad_(False)
        self.gate.requires_grad_(True)
        return self

    def unfreeze_experts(self) -> "GatedDualExpert":
        self.physics_calibrator.requires_grad_(True)
        self.ai_expert.requires_grad_(True)
        return self

    def parameter_counts(self) -> Dict[str, int]:
        groups = {
            "physics_calibration": self.physics_calibrator,
            "ai_expert": self.ai_expert,
            "reliability_gate": self.gate,
        }
        counts = {name: sum(p.numel() for p in module.parameters()) for name, module in groups.items()}
        counts["total"] = sum(counts.values())
        counts["trainable_total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts


def gate_target(
    p_physics: ArrayLike,
    p_ai: ArrayLike,
    labels: ArrayLike,
    *,
    temperature: float = 1.0,
    eps: float = _EPS,
) -> Tensor:
    """Soft target for the gate: values above .5 mean AI has lower BCE loss.

    ``sigmoid((loss_physics - loss_ai) / temperature)`` is detached from both
    experts, so gate supervision cannot update an expert through its target.
    """
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    p_p = _check_probability(p_physics, "p_physics")
    p_a = _check_probability(p_ai, "p_ai")
    y = _check_labels(labels)
    _same_rows(("p_physics", p_p), ("p_ai", p_a), ("labels", y))
    with torch.no_grad():
        loss_physics = _safe_bce(p_p, y, eps)
        loss_ai = _safe_bce(p_a, y, eps)
        return torch.sigmoid((loss_physics - loss_ai) / float(temperature))


def gated_loss(
    output: GatedOutput,
    labels: ArrayLike,
    *,
    lambda_g: float = 0.5,
    lambda_p: float = 0.05,
    gate_temperature: float = 1.0,
    eps: float = _EPS,
    return_terms: bool = False,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
    """Compute fusion BCE + gate BCE + physics-respect regularization.

    The physics-respect term penalizes AI takeover only where the calibrated
    physics expert is both correct and confident.  It does not force the gate
    to prefer physics where the physics score is mismatched to the corruption.
    """
    if not isinstance(output, GatedOutput):
        raise TypeError("output must be GatedOutput from GatedDualExpert.components")
    for name, value in (("p_physics", output.p_physics), ("p_ai", output.p_ai),
                        ("alpha", output.alpha), ("p_fused", output.p_fused)):
        _check_probability(value, name)
    y = _check_labels(labels)
    _same_rows(("p_fused", output.p_fused), ("labels", y))
    if not math.isfinite(float(lambda_g)) or lambda_g < 0:
        raise ValueError("lambda_g must be finite and non-negative")
    if not math.isfinite(float(lambda_p)) or lambda_p < 0:
        raise ValueError("lambda_p must be finite and non-negative")

    fusion_bce = _safe_bce(output.p_fused, y, eps).mean()
    target = gate_target(
        output.p_physics, output.p_ai, y, temperature=gate_temperature, eps=eps
    )
    gate_bce = _safe_bce(output.alpha, target, eps).mean()
    with torch.no_grad():
        physics_correct = ((output.p_physics >= 0.5) == (y >= 0.5)).to(torch.float32)
        physics_confidence = torch.abs(2.0 * output.p_physics - 1.0)
    physics_respect = (physics_confidence * physics_correct * output.alpha).mean()
    total = fusion_bce + float(lambda_g) * gate_bce + float(lambda_p) * physics_respect
    if not torch.isfinite(total).item():
        raise FloatingPointError("non-finite gated loss")
    if not return_terms:
        return total
    return total, {
        "fusion_bce": fusion_bce,
        "gate_bce": gate_bce,
        "physics_respect": physics_respect,
        "gate_target_mean": target.mean(),
    }


def validation_threshold(
    validation_probabilities: ArrayLike,
    validation_labels: ArrayLike,
    *,
    target_source_recall: float = 0.98,
) -> float:
    """Choose a fixed threshold from validation positives only.

    This intentionally names its input ``validation_*``: held-out event,
    EBIV, and EBOS results must never tune the selection operating point.
    """
    if not math.isfinite(float(target_source_recall)) or not 0 < target_source_recall <= 1:
        raise ValueError("target_source_recall must lie in (0, 1]")
    probabilities = _check_probability(validation_probabilities, "validation_probabilities")
    labels = _check_labels(validation_labels, "validation_labels")
    _same_rows(("validation_probabilities", probabilities), ("validation_labels", labels))
    positives = probabilities.detach().cpu().numpy().reshape(-1)[labels.detach().cpu().numpy().reshape(-1) > 0.5]
    if len(positives) == 0:
        raise InputValidationError("validation set contains no source proxy-positive events")
    return float(np.quantile(positives, 1.0 - float(target_source_recall), method="lower"))


def threshold_mask(probabilities: ArrayLike, threshold: float) -> np.ndarray:
    """Return a boolean event-retention mask for a previously frozen threshold."""
    if not math.isfinite(float(threshold)) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    probabilities_t = _check_probability(probabilities, "probabilities")
    return (probabilities_t.detach().cpu().numpy().reshape(-1) >= float(threshold))


def predict_probabilities(
    model: GatedDualExpert,
    generic_features: ArrayLike,
    physics_features: ArrayLike,
    physics_scores: ArrayLike,
    *,
    batch_size: int = 65536,
) -> np.ndarray:
    """CPU batched inference returning fused probabilities as a NumPy vector."""
    if not isinstance(model, GatedDualExpert):
        raise TypeError("model must be GatedDualExpert")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    generic = _as_features(generic_features, "generic_features", AIExpert.input_dim)
    physics = _as_features(physics_features, "physics_features", 8)
    scores = _as_column(physics_scores, "physics_scores")
    _same_rows(("generic_features", generic), ("physics_features", physics), ("physics_scores", scores))
    was_training = model.training
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, generic.shape[0], batch_size):
            stop = min(start + batch_size, generic.shape[0])
            values.append(model(generic[start:stop], physics[start:stop], scores[start:stop]).reshape(-1))
    if was_training:
        model.train()
    if not values:
        return np.empty(0, dtype=np.float32)
    return torch.cat(values).cpu().numpy()


__all__ = [
    "AIExpert",
    "GatedDualExpert",
    "GatedOutput",
    "InputValidationError",
    "PhysicsCalibrator",
    "ReliabilityGate",
    "fit_platt",
    "gate_target",
    "gated_loss",
    "make_gate_features",
    "predict_probabilities",
    "set_cpu_seed",
    "threshold_mask",
    "validation_threshold",
]
