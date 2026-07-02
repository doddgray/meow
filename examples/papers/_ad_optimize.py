"""Shared AD-gradient-based optimizer for the designer examples.

A minimal `Adam <https://arxiv.org/abs/1412.6980>`_ optimizer driven by
``jax.grad`` of a loss built from one of meow's differentiable EME primitives
(``make_differentiable_neffs``, ``make_differentiable_modes`` or
``make_differentiable_objective`` - this module is agnostic to which; it only
needs a ``jax``-differentiable ``loss_fn(params) -> scalar``). Records a
per-iteration trace (loss, parameters, gradient norm) for the optimization-trace
plots the designer examples show alongside their optimized layout/performance.

This is intentionally small (no external optimizer dependency): box-constrained
Adam is enough for the low-dimensional (2-4 parameter), well-conditioned
objectives these designers optimize (phase-matching, coupling contrast).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import matplotlib.pyplot as plt


@dataclass
class OptimizationTrace:
    """Per-iteration record of an :func:`adam_minimize` run."""

    objective_name: str = "objective"
    param_names: tuple[str, ...] = ()
    losses: list[float] = field(default_factory=list)
    params: list[np.ndarray] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(losses, params, grad_norms)`` as stacked arrays."""
        return (
            np.asarray(self.losses),
            np.asarray(self.params),
            np.asarray(self.grad_norms),
        )


def adam_minimize(
    loss_fn: Callable[[Any], Any],
    x0: Sequence[float],
    *,
    steps: int = 40,
    lr: float = 0.05,
    bounds: Sequence[tuple[float, float]] | None = None,
    param_names: tuple[str, ...] = (),
    objective_name: str = "loss",
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Minimize ``loss_fn(params)`` via ``jax.grad`` + Adam, with box bounds.

    Args:
        loss_fn: a ``jax``-differentiable ``params -> scalar`` to minimize (build
            it from ``meow.make_differentiable_neffs`` / ``make_differentiable_modes``
            / ``make_differentiable_objective`` so the gradient is real).
        x0: the initial parameter vector.
        steps: number of Adam iterations.
        lr: Adam learning rate.
        bounds: optional per-parameter ``(low, high)`` box constraints, applied
            by clipping after each step (projected gradient).
        param_names: names for the trace/plots (defaults to ``p0, p1, ...``).
        objective_name: label for the trace/plots.
        beta1: Adam first-moment decay rate.
        beta2: Adam second-moment decay rate.
        eps: Adam denominator stabilizer.

    Returns:
        ``(x_opt, trace)`` - the final parameter vector and the optimization
        trace (loss/params/grad-norm per iteration, including the initial point).
    """
    import jax
    import jax.numpy as jnp

    x = jnp.asarray(x0, dtype=float)
    names = param_names or tuple(f"p{i}" for i in range(x.size))
    lo = jnp.array([b[0] for b in bounds]) if bounds else None
    hi = jnp.array([b[1] for b in bounds]) if bounds else None
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)
    trace = OptimizationTrace(objective_name=objective_name, param_names=names)

    value_and_grad = jax.value_and_grad(loss_fn)
    for t in range(1, steps + 1):
        loss, grad = value_and_grad(x)
        trace.losses.append(float(loss))
        trace.params.append(np.asarray(x))
        trace.grad_norms.append(float(jnp.linalg.norm(grad)))
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        mhat = m / (1 - beta1**t)
        vhat = v / (1 - beta2**t)
        x = x - lr * mhat / (jnp.sqrt(vhat) + eps)
        if bounds is not None:
            x = jnp.clip(x, lo, hi)
    # record the converged point too, for a trace that ends where x_opt is
    trace.losses.append(float(loss_fn(x)))
    trace.params.append(np.asarray(x))
    trace.grad_norms.append(trace.grad_norms[-1])
    return np.asarray(x), trace


def adam_maximize(
    objective_fn: Callable[[Any], Any], x0: Sequence[float], **kwargs: Any
) -> tuple[np.ndarray, OptimizationTrace]:
    """Adam ascent: maximize ``objective_fn`` (minimizes its negation)."""
    kwargs.setdefault("objective_name", "objective")

    def neg(params: Any) -> Any:
        return -objective_fn(params)

    x_opt, trace = adam_minimize(neg, x0, **kwargs)
    trace.losses = [-loss for loss in trace.losses]  # report the objective, not -it
    return x_opt, trace


def plot_trace(
    trace: OptimizationTrace,
    ax_loss: plt.Axes,
    ax_params: plt.Axes,
    *,
    loss_ylog: bool = False,
) -> None:
    """Plot the objective and parameter trajectories vs. iteration."""
    losses, params, _grad_norms = trace.as_arrays()
    it = np.arange(len(losses))
    ax_loss.plot(it, losses, "C0o-", ms=3)
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel(trace.objective_name)
    if loss_ylog:
        ax_loss.set_yscale("log")
    ax_loss.grid(visible=True)
    ax_loss.set_title(f"AD optimization trace: {trace.objective_name}")

    for j, name in enumerate(trace.param_names):
        ax_params.plot(it, params[:, j], "o-", ms=3, label=name)
    ax_params.set_xlabel("iteration")
    ax_params.set_ylabel("parameter value")
    ax_params.legend(fontsize=8)
    ax_params.grid(visible=True)
    ax_params.set_title("design parameters")
