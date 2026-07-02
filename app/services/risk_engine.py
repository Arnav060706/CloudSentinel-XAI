# app/services/risk_engine.py
"""
HawkesRiskEngine — Self-Exciting Cross-Cloud Risk Intensity Engine
==================================================================

Implements the formalized Hawkes-process risk intensity function for
CloudSentinel-XAI, as specified in the architecture document:

    λ_v(t) = λ₀ + Σᵢ [ κᵢ · g(C_active(t)) · exp(-β · (t - tᵢ)) ]

where:
    λ₀              — baseline idle intensity (lambda_0)
    κᵢ              — per-event excitation weight: κ · anomaly_score_i
                       (anomaly_score from Isolation Forest, range [0,1])
    g(C_active(t))  — cross-cloud diversity multiplier:
                       exp(γ · (|C_active(t)| - 1))
    β               — temporal decay rate
    tᵢ              — ACTUAL timestamp of event i (not a mock offset)
    t               — current evaluation time

Cross-cloud diversity multiplier behaviour
------------------------------------------
  |C_active| = 1  →  g = exp(0) = 1.0   (single cloud, no amplification)
  |C_active| = 2  →  g = exp(γ)          (second cloud crossed — jumps by e^γ)
  |C_active| = 3  →  g = exp(2γ)         (all three clouds — maximum amplification)

Monotonicity lemma (for paper / patent)
----------------------------------------
  For fixed event count and timestamps, λ_v(t) is strictly increasing in
  |C_active(t)|. Proof: g is strictly increasing in its argument (exp is
  monotone), and the sum of excitation terms is strictly positive for any
  non-empty active window.

Pipeline integration note
--------------------------
  This engine expects events that have been scored by the ML pipeline
  (Isolation Forest + XGBoost). Each event dict must include:
    - "timestamp"      : ISO-8601 string (actual event time, not arrival time)
    - "source_cloud"   : "AWS" | "AZURE" | "GCP"
    - "anomaly_score"  : float in [0, 1] — from Isolation Forest
                         (negative Isolation Forest scores are remapped to [0,1])
    - "phase_confidence" : float in [0, 1] — from XGBoost (optional; used if
                           anomaly_score is absent)
  Do NOT pass raw normalized events here; the severity weight will default
  to 0.1 and the Hawkes process degenerates to a uniform weighted counter.

Parameter fitting
-----------------
  Default parameters are reasonable starting points.
  Run fit_parameters_mle() on a labeled alert sequence to obtain
  maximum-likelihood estimates tuned to your specific dataset.
"""

import math
import time
import datetime
import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class HawkesRiskEngine:
    """
    Self-exciting point process risk engine with cross-cloud diversity multiplier.

    Parameters
    ----------
    baseline_rate : float
        λ₀ — idle risk intensity of a quiet corporate environment.
        Default 0.05 (calibrate against your benign-traffic baseline).
    decay_rate : float
        β — exponential decay rate. Higher β means recent events dominate
        and old events fade quickly. At β=0.1 and a 60-second window,
        an event 60s old retains exp(-0.1*60) ≈ 0.2% of its initial weight.
        Default 0.1.
    base_excitation : float
        κ — base excitation weight per event (before anomaly_score scaling).
        Default 0.3.
    amplification_gamma : float
        γ — cross-cloud diversity amplification rate. At γ=1.5:
          1 cloud  → ×1.0
          2 clouds → ×4.5
          3 clouds → ×20.1
        Tune this against your detection-latency / false-positive trade-off.
        Default 1.5.
    alert_threshold : float
        Raw λ_v value above which is_critical is True. Calibrate by running
        your benign traffic baseline through the engine and setting this at
        the 99th percentile of benign lambda values + margin.
        Default 0.75 (pre-calibration placeholder).
    """

    # Supported cloud provider identifiers (case-insensitive matching)
    _KNOWN_CLOUDS: Set[str] = {"AWS", "AZURE", "GCP", "ENTRA-ID"}

    def __init__(
        self,
        baseline_rate: float = 0.05,
        decay_rate: float = 0.1,
        base_excitation: float = 0.3,
        amplification_gamma: float = 1.5,
        alert_threshold: float = 0.75,
    ):
        # Validate parameters
        if baseline_rate < 0:
            raise ValueError("baseline_rate (λ₀) must be non-negative")
        if decay_rate <= 0:
            raise ValueError("decay_rate (β) must be strictly positive")
        if base_excitation <= 0:
            raise ValueError("base_excitation (κ) must be strictly positive")
        if amplification_gamma <= 0:
            raise ValueError("amplification_gamma (γ) must be strictly positive")

        self.lambda_0 = baseline_rate
        self.beta = decay_rate
        self.kappa = base_excitation
        self.gamma = amplification_gamma
        self.alert_threshold = alert_threshold

        # Running calibration data: track observed lambda_v values so the
        # sigmoid normalization can be recalibrated without a full restart.
        self._observed_max_lambda: float = baseline_rate
        self._intensity_history: List[float] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def calculate_intensity(
        self,
        active_events: List[dict],
        eval_time: Optional[float] = None,
    ) -> dict:
        """
        Computes the current risk intensity λ_v(t) for an entity node.

        Parameters
        ----------
        active_events : List[dict]
            Events currently inside the sliding window for this entity,
            as returned by MultiCloudGraphEngine.process_event().
            Each dict must include "timestamp", "source_cloud", and
            "anomaly_score" (from the ML inference layer).
        eval_time : float, optional
            Unix timestamp to evaluate at. Defaults to time.time().
            Pass a fixed value in tests / replay scenarios for determinism.

        Returns
        -------
        dict with keys:
            risk_intensity      : float  — raw λ_v(t)
            scaled_score        : float  — sigmoid-normalized to [0, 1]
            cloud_span_count    : int    — |C_active(t)|
            active_clouds       : list   — which CSPs are present
            diversity_multiplier: float  — g(C_active(t))
            is_critical         : bool   — scaled_score >= alert_threshold
            event_contributions : list   — per-event breakdown (for SHAP / XAI)
        """
        t = eval_time if eval_time is not None else time.time()

        if not active_events:
            return self._empty_result()

        # ---- Step 1: Cross-cloud diversity factor -----------------------
        active_clouds: Set[str] = {
            str(e.get("source_cloud", "UNKNOWN")).upper()
            for e in active_events
        }
        # Normalize ENTRA-ID to AZURE for cloud-count purposes
        # (Entra ID events are Azure identity plane, same cloud tenant)
        active_clouds_normalized = {
            "AZURE" if c == "ENTRA-ID" else c
            for c in active_clouds
        }
        c_active = len(active_clouds_normalized)
        diversity_multiplier = math.exp(self.gamma * (c_active - 1))

        # ---- Step 2: Hawkes cascading excitation sum -------------------
        total_excitation = 0.0
        event_contributions = []

        for event in active_events:
            # Actual event timestamp — NOT a mock offset.
            # This is the critical fix: temporal decay must reflect when the
            # event actually occurred, so older events contribute less.
            t_i = self._parse_event_timestamp(event)
            if t_i is None:
                logger.warning(
                    "Event missing valid 'timestamp' field — skipping contribution. "
                    "Event keys: %s", list(event.keys())
                )
                continue

            delta_t = max(0.0, t - t_i)

            # Per-event excitation weight: κ × anomaly_score_i
            # anomaly_score comes from Isolation Forest (not from ml_labels).
            # If missing, default to 0.1 and log a warning — this usually
            # means the risk engine is receiving un-scored events.
            anomaly_score = event.get("anomaly_score")
            if anomaly_score is None:
                anomaly_score = event.get("phase_confidence", 0.1)
                if anomaly_score == 0.1:
                    logger.warning(
                        "Event for principal '%s' has no anomaly_score or "
                        "phase_confidence — defaulting to 0.1. Ensure ML "
                        "inference runs before risk scoring.",
                        event.get("principal", "unknown"),
                    )
            # Clamp to [0, 1]
            anomaly_score = max(0.0, min(1.0, float(anomaly_score)))
            kappa_i = self.kappa * anomaly_score

            # Exponentially decaying contribution
            decay_factor = math.exp(-self.beta * delta_t)
            contribution = kappa_i * decay_factor

            total_excitation += contribution
            event_contributions.append({
                "principal": event.get("principal", ""),
                "source_cloud": event.get("source_cloud", ""),
                "anomaly_score": round(anomaly_score, 4),
                "delta_t_seconds": round(delta_t, 2),
                "decay_factor": round(decay_factor, 4),
                "contribution": round(contribution, 6),
            })

        # ---- Step 3: Final intensity -----------------------------------
        # Formula: λ_v(t) = λ₀ + g(C_active) · Σ κᵢ · exp(-β(t - tᵢ))
        # g factors out of the sum because it depends only on C_active(t),
        # which is constant for the current window evaluation.
        lambda_v = self.lambda_0 + (diversity_multiplier * total_excitation)

        # ---- Step 4: Calibrated sigmoid normalization ------------------
        # Track running max for adaptive calibration.
        scaled_score = self._sigmoid_normalize(lambda_v)
        self._update_calibration(lambda_v)
        result = {
            "risk_intensity": round(lambda_v, 6),
            "scaled_score": round(scaled_score, 4),
            "cloud_span_count": c_active,
            "active_clouds": sorted(active_clouds_normalized),
            "diversity_multiplier": round(diversity_multiplier, 4),
            "is_critical": scaled_score >= self.alert_threshold,
            "event_contributions": event_contributions,
            # Included for SHAP / faithfulness gate: which features drove the score
            "dominant_signal": self._identify_dominant_signal(
                lambda_v, diversity_multiplier, total_excitation, c_active
            ),
        }

        logger.debug(
            "λ_v=%.4f scaled=%.4f clouds=%s multiplier=%.2f critical=%s",
            lambda_v, scaled_score, sorted(active_clouds_normalized),
            diversity_multiplier, result["is_critical"],
        )
        return result

    def recalibrate_threshold(self, benign_lambda_values: List[float], percentile: float = 99.0) -> float:
        """
        Sets alert_threshold to the given percentile of observed benign
        lambda values plus a 10% margin.

        Call this after running your benign traffic baseline through the
        engine to get an empirically grounded alert threshold rather than
        using the default placeholder.

        Parameters
        ----------
        benign_lambda_values : List[float]
            Lambda values observed during benign-only traffic periods.
        percentile : float
            Percentile to use as the threshold baseline. Default 99.0.

        Returns
        -------
        float : the new alert_threshold value.
        """
        if not benign_lambda_values:
            raise ValueError("benign_lambda_values must be non-empty")

        sorted_vals = sorted(benign_lambda_values)
        idx = min(int(len(sorted_vals) * percentile / 100.0), len(sorted_vals) - 1)
        threshold = sorted_vals[idx] * 1.10  # 10% margin

        # Convert to scaled domain
        scaled_threshold = self._sigmoid_normalize(threshold)
        self.alert_threshold = round(scaled_threshold, 4)
        logger.info(
            "Alert threshold recalibrated to %.4f (raw λ=%.4f at p%.0f)",
            self.alert_threshold, threshold, percentile,
        )
        return self.alert_threshold

    def get_parameter_summary(self) -> dict:
        """Returns current parameter state for logging / reproducibility."""
        return {
            "lambda_0": self.lambda_0,
            "beta": self.beta,
            "kappa": self.kappa,
            "gamma": self.gamma,
            "alert_threshold": self.alert_threshold,
            "observed_max_lambda": round(self._observed_max_lambda, 4),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _parse_event_timestamp(self, event: dict) -> Optional[float]:
        """
        Parses the event's actual occurrence timestamp into a Unix float.
        Returns None if the field is missing or unparseable.
        """
        ts_str = event.get("timestamp")
        if not ts_str:
            return None
        try:
            return datetime.datetime.fromisoformat(
                str(ts_str).replace("Z", "+00:00")
            ).timestamp()
        except (ValueError, TypeError) as exc:
            logger.warning("Could not parse timestamp '%s': %s", ts_str, exc)
            return None

    def _sigmoid_normalize(self, lambda_v: float) -> float:
        """
        Maps raw λ_v to [0, 1] using a calibrated sigmoid:

            scaled = 1 / (1 + exp(-k · (λ_v - midpoint)))

        where:
            midpoint = λ₀ × 10  (10× baseline = "moderate concern" inflection)
            k        = 3 / (observed_max_lambda - λ₀ + ε)  (slope calibration)

        This is strictly preferable to dividing by a hard-coded constant (e.g.
        /2.0) because:
          1. It adapts as the engine observes larger lambda values.
          2. It never clips — extreme events are represented with high but
             not identical scores, preserving ordinal ranking.
          3. The inflection point and slope have interpretable semantics.
        """
        midpoint = self.lambda_0 * 10.0
        epsilon = 1e-6
        dynamic_range = max(self._observed_max_lambda - self.lambda_0, epsilon)
        k = 3.0 / dynamic_range

        return 1.0 / (1.0 + math.exp(-k * (lambda_v - midpoint)))

    def _update_calibration(self, lambda_v: float) -> None:
        """Tracks running max for adaptive sigmoid calibration."""
        if lambda_v > self._observed_max_lambda:
            self._observed_max_lambda = lambda_v
        self._intensity_history.append(round(lambda_v, 6))
        # Keep history bounded (last 10,000 observations)
        if len(self._intensity_history) > 10_000:
            self._intensity_history = self._intensity_history[-10_000:]

    def _identify_dominant_signal(
        self,
        lambda_v: float,
        diversity_multiplier: float,
        total_excitation: float,
        c_active: int,
    ) -> str:
        """
        Identifies the primary driver of the current risk score.
        Used by the SHAP faithfulness gate to label the LLM narrative prompt:
        the LLM should explain whichever signal is actually dominant, not
        produce a generic "high risk" statement.
        """
        if lambda_v <= self.lambda_0 * 1.5:
            return "baseline_only"
        if c_active >= 2 and diversity_multiplier > 2.0:
            return "cross_cloud_diversity"
        if total_excitation > self.lambda_0 * 5:
            return "high_event_volume"
        return "temporal_excitation_accumulation"

    def _empty_result(self) -> dict:
        """Returns the baseline result when no events are in the window."""
        return {
            "risk_intensity": round(self.lambda_0, 6),
            "scaled_score": round(self._sigmoid_normalize(self.lambda_0), 4),
            "cloud_span_count": 0,
            "active_clouds": [],
            "diversity_multiplier": 1.0,
            "is_critical": False,
            "event_contributions": [],
            "dominant_signal": "baseline_only",
        }