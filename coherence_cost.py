"""
Coherence Cost (S_coh) Calculator for Boundary-Driven CA
========================================================

This module computes explicit coherence cost metrics that can be integrated
into the CA simulation to validate the TRA theoretical bridge.

Three operational definitions of S_coh:
1. Structural KL divergence: how much the tracked structure changed
2. Predictive surprise: mismatch between predicted and actual next state
3. Memory alignment cost: distance from memory halo

Author: Erez Ashkenazi (with Claude assistance)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


class CoherenceCostCalculator:
    """
    Computes coherence cost S_coh for observer-tracked structures in CA.

    The coherence cost quantifies how much the tracked structure deviates
    from the observer's internal constraints (memory, predictions).
    """

    def __init__(self, window_size: tuple[int, int], memory_depth: int = 5, prediction_horizon: int = 1):
        self.window_size = window_size
        self.memory_depth = memory_depth
        self.prediction_horizon = prediction_horizon

        self.memory_buffer: list[np.ndarray] = []

        self.S_coh_history: list[float] = []
        self.S_pred_history: list[float] = []
        self.S_total_history: list[float] = []

    def update(
        self,
        current_core_mask: np.ndarray,
        predicted_core_mask: np.ndarray | None = None,
        observer_mode: str = "passive",
        intervention_mask: np.ndarray | None = None,
        *,
        eta: float = 1.0,
        w1: float = 0.5,
        w2: float = 0.3,
        w3: float = 0.2,
        epsilon: float = 1e-6,
    ) -> tuple[float, float, float]:
        """
        Compute and record coherence cost for current timestep.
        Returns: (S_coh, S_pred, S_total)
        """
        C = current_core_mask.astype(float)

        self.memory_buffer.append(C.copy())
        if len(self.memory_buffer) > self.memory_depth:
            self.memory_buffer.pop(0)

        S_coh = self._compute_coherence_cost(
            C, observer_mode, intervention_mask, w1=w1, w2=w2, w3=w3, epsilon=epsilon
        )
        S_pred = self._compute_predictive_surprisal(C, predicted_core_mask, epsilon=epsilon)
        S_total = float(S_pred + eta * S_coh)

        self.S_coh_history.append(float(S_coh))
        self.S_pred_history.append(float(S_pred))
        self.S_total_history.append(float(S_total))
        return float(S_coh), float(S_pred), float(S_total)

    def _compute_coherence_cost(
        self,
        current_mask: np.ndarray,
        observer_mode: str,
        intervention_mask: np.ndarray | None,
        *,
        w1: float,
        w2: float,
        w3: float,
        epsilon: float,
    ) -> float:
        if len(self.memory_buffer) < 2:
            return 0.0

        prev_mask = self.memory_buffer[-2]
        S_structural = self._kl_divergence_symmetric(prev_mask, current_mask, epsilon=epsilon)

        if "persist" in observer_mode:
            S_memory = self._memory_alignment_cost(current_mask, epsilon=epsilon)
        else:
            S_memory = 0.0

        if intervention_mask is not None:
            S_intervention = self._intervention_alignment_cost(intervention_mask, current_mask, epsilon=epsilon)
        else:
            S_intervention = 0.0

        return float(w1 * S_structural + w2 * S_memory + w3 * S_intervention)

    def _compute_predictive_surprisal(
        self,
        current_mask: np.ndarray,
        predicted_mask: np.ndarray | None,
        *,
        epsilon: float,
    ) -> float:
        if predicted_mask is None:
            if len(self.memory_buffer) < 2:
                return 0.0
            predicted_mask = self.memory_buffer[-2]

        error = np.abs(current_mask - predicted_mask).sum()
        total_cells = current_mask.size
        error_rate = float(error / max(total_cells, 1))
        return float(-np.log(1.0 - error_rate + epsilon))

    def _kl_divergence_symmetric(self, mask1: np.ndarray, mask2: np.ndarray, *, epsilon: float) -> float:
        p1 = mask1.flatten().astype(float)
        p2 = mask2.flatten().astype(float)

        p1 = (p1 + epsilon) / (p1.sum() + epsilon * len(p1))
        p2 = (p2 + epsilon) / (p2.sum() + epsilon * len(p2))

        kl_12 = float(np.sum(p1 * np.log(p1 / p2)))
        kl_21 = float(np.sum(p2 * np.log(p2 / p1)))
        return float((kl_12 + kl_21) / 2.0)

    def _memory_alignment_cost(self, current_mask: np.ndarray, *, epsilon: float) -> float:
        if len(self.memory_buffer) < 2:
            return 0.0

        halo = np.zeros_like(current_mask, dtype=bool)
        for past_mask in self.memory_buffer[:-1]:
            dilated = binary_dilation(past_mask.astype(bool), iterations=2)
            halo = halo | dilated

        overlap = np.logical_and(current_mask.astype(bool), halo).sum()
        current_size = current_mask.sum()

        if current_size <= 0:
            return 1.0

        alignment = float(overlap / (current_size + epsilon))
        return float(1.0 - alignment)

    def _intervention_alignment_cost(self, intervention_mask: np.ndarray, current_mask: np.ndarray, *, epsilon: float) -> float:
        if intervention_mask.sum() == 0:
            return 0.0

        misaligned = np.logical_and(intervention_mask.astype(bool), np.logical_not(current_mask.astype(bool))).sum()
        total_interventions = intervention_mask.sum()
        return float(misaligned / (total_interventions + epsilon))

    def get_adequacy_proxy(self, lambda_coupling: float = 1.0) -> list[float]:
        return [float(np.exp(-lambda_coupling * S)) for S in self.S_total_history]

    def get_temporal_distortion(
        self, tau_baseline: float = 1.0, lambda_coupling: float = 1.0
    ) -> tuple[list[float], list[float]]:
        D_history = [float(np.exp(lambda_coupling * S)) for S in self.S_total_history]
        t_effective_history = [float(tau_baseline * D) for D in D_history]
        return D_history, t_effective_history

    def correlate_with_identity_score(self, identity_scores: list[float]) -> float:
        A_history = self.get_adequacy_proxy()
        if len(A_history) != len(identity_scores):
            raise ValueError("Mismatched history lengths")
        return float(np.corrcoef(A_history, identity_scores)[0, 1])

    def plot_summary(self, identity_scores: list[float] | None = None, save_path: str | None = None):
        import matplotlib.pyplot as plt

        steps = range(len(self.S_coh_history))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        ax = axes[0, 0]
        ax.plot(steps, self.S_coh_history, label=r"$S_{coh}$", alpha=0.7)
        ax.plot(steps, self.S_pred_history, label=r"$S_{pred}$", alpha=0.7)
        ax.plot(steps, self.S_total_history, label=r"$S_{total}$", linewidth=2)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Misfit")
        ax.set_title("Coherence and Predictive Costs")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        A_history = self.get_adequacy_proxy()
        ax.plot(steps, A_history, linewidth=2)
        ax.set_xlabel("Time step")
        ax.set_ylabel(r"Adequacy $\hat{A}$")
        ax.set_title("Observer Adequacy (TRA)")
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        D_history, _t_eff = self.get_temporal_distortion()
        ax.plot(steps, D_history, linewidth=2)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Distortion D")
        ax.set_title("Temporal Distortion")
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        if identity_scores is not None:
            ax.scatter(A_history, identity_scores, alpha=0.5)
            ax.set_xlabel(r"Adequacy $\hat{A}$")
            ax.set_ylabel("Identity Score")
            corr = self.correlate_with_identity_score(identity_scores)
            ax.set_title(f"TRA Bridge Test (r={corr:.3f})")
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No identity scores provided", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Identity Score Comparison")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig
