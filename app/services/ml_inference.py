"""
ParallelMLEngine — Asynchronous Multi-Model Inference & XAI Extraction
======================================================================

Layer 3 inference: Isolation Forest anomaly score + XGBoost ATT&CK phase
classification + TreeSHAP attributions. CPU-bound work is wrapped in
asyncio.to_thread and run concurrently.

Bypass mode
-----------
If trained models are not present in the shared state_matrix (e.g. before
the dataset is generated and the models are trained), every inference call
returns neutral, well-formed defaults so the rest of the pipeline still
runs end to end. There is exactly ONE bypass exit and it always sets the
same keys, so no downstream consumer can hit a missing field.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class ParallelMLEngine:
    """Non-blocking, parallel ML inference reading models from state_matrix."""

    # Feature schema (X). Must exclude target/leakage columns.
    _ORDERED_FEATURES = [
        "source_cloud",
        "principal_type",
        "principal_created_in_window",
        "is_known_proxy_or_tor",
        "ip_geo_country",
        "ua_family",
        "ua_version",
        "action",
        "status",
    ]

    def __init__(self, state_matrix: dict):
        self.iso_forest = state_matrix.get("iso_forest")
        self.xgboost_model = state_matrix.get("xgboost")
        self.feature_encoder = state_matrix.get("feature_encoder")
        # Optional persisted index->phase-name mapping.
        self.class_names = state_matrix.get("class_names")

        self._models_ready = bool(self.iso_forest and self.xgboost_model)
        if not self._models_ready:
            logger.warning(
                "ParallelMLEngine initialized without trained models — "
                "running in BYPASS mode (neutral inference defaults)."
            )

        # TreeSHAP explainer, only if XGBoost is present.
        self.explainer = None
        if SHAP_AVAILABLE and self.xgboost_model is not None:
            try:
                self.explainer = shap.TreeExplainer(self.xgboost_model)
            except Exception as e:
                logger.error("Failed to build TreeExplainer: %s", e)
                self.explainer = None

    # ------------------------------------------------------------------ #
    # Public async API                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bypass(event: dict) -> dict:
        """Single, consistent neutral-default exit for bypass / error paths."""
        event["anomaly_score"] = 0.1
        event["predicted_phase"] = "Normal"
        event["predicted_phase_index"] = 0
        event["phase_confidence"] = 0.1
        event["shap_attributions"] = {}
        return event

    async def execute_parallel_inference(self, normalized_event: dict) -> dict:
        if not self._models_ready:
            return self._bypass(normalized_event)

        # 1. Prepare features
        try:
            x_tensor, raw_df = self._prepare_feature_vector(normalized_event)
        except Exception as e:
            logger.error("Feature encoding failure: %s", e)
            return self._bypass(normalized_event)

        # 2. Run models concurrently in threads
        try:
            iso_task = asyncio.to_thread(self._run_isolation_forest, x_tensor)
            xgb_task = asyncio.to_thread(self._run_xgboost_and_shap, x_tensor, raw_df)
            iso_result, xgb_result = await asyncio.gather(iso_task, xgb_task)
        except Exception as e:
            logger.error("Inference failure: %s", e)
            return self._bypass(normalized_event)

        # 3. Enrich the event
        normalized_event["anomaly_score"] = iso_result
        normalized_event["predicted_phase"] = xgb_result.get("predicted_phase", "Normal")
        normalized_event["predicted_phase_index"] = xgb_result.get("predicted_phase_index", 0)
        normalized_event["phase_confidence"] = xgb_result.get("confidence", 0.0)
        normalized_event["shap_attributions"] = xgb_result.get("shap_attributions", {})
        return normalized_event

    # ------------------------------------------------------------------ #
    # Internal threaded workers                                          #
    # ------------------------------------------------------------------ #

    def _prepare_feature_vector(self, event: dict) -> Tuple[Any, pd.DataFrame]:
        raw_features = {feat: event.get(feat, "Unknown") for feat in self._ORDERED_FEATURES}
        df = pd.DataFrame([raw_features])

        if self.feature_encoder is not None:
            encoded_tensor = self.feature_encoder.transform(df)
            return encoded_tensor, df

        # No fitted encoder: sklearn models cannot consume raw strings. Rather
        # than crash mid-inference, signal the caller to fall back to bypass.
        raise RuntimeError(
            "No feature_encoder available; cannot build a numeric tensor for "
            "the sklearn/XGBoost models. Provide models/feature_encoder.pkl."
        )

    def _run_isolation_forest(self, x_tensor) -> float:
        raw_score = self.iso_forest.decision_function(x_tensor)[0]
        # Lower (more negative) => more anomalous. Remap to [0,1] where 1 is
        # highly anomalous. decision_function is not guaranteed to sit in
        # [-0.5, 0.5]; clamp defensively. (For a calibrated mapping, fit an
        # empirical CDF over benign scores — noted for the eval phase.)
        normalized_score = 0.5 - raw_score
        return round(max(0.0, min(1.0, normalized_score)), 4)

    def _run_xgboost_and_shap(self, x_tensor, raw_df: pd.DataFrame) -> Dict[str, Any]:
        probabilities = np.asarray(self.xgboost_model.predict_proba(x_tensor)[0])
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        predicted_phase = self._phase_name(predicted_index)

        top_shap_features: Dict[str, Any] = {}
        if self.explainer is not None:
            try:
                top_shap_features = self._extract_shap(x_tensor, raw_df, predicted_index)
            except Exception as e:
                logger.error("SHAP extraction failed: %s", e)
                top_shap_features = {}

        return {
            "predicted_phase": str(predicted_phase),
            "predicted_phase_index": predicted_index,
            "confidence": round(confidence, 4),
            "shap_attributions": top_shap_features,
        }

    def _phase_name(self, index: int) -> str:
        # Prefer an explicit persisted mapping (index -> ATT&CK phase name).
        if self.class_names and index < len(self.class_names):
            return str(self.class_names[index])
        # Otherwise fall back to the model's classes_ (often integer labels).
        classes = getattr(self.xgboost_model, "classes_", None)
        if classes is not None and index < len(classes):
            return str(classes[index])
        return "Unknown"

    def _extract_shap(self, x_tensor, raw_df: pd.DataFrame, predicted_index: int) -> Dict[str, Any]:
        """
        Robust to both SHAP return conventions:
          - legacy list-of-arrays: shap_values[class][sample]
          - modern ndarray (n_samples, n_features) binary, or
            (n_samples, n_features, n_classes) multiclass.
        """
        shap_values = self.explainer.shap_values(x_tensor)

        if isinstance(shap_values, list):
            # list indexed by class -> array (n_samples, n_features)
            idx = predicted_index if predicted_index < len(shap_values) else 0
            class_shap_values = np.asarray(shap_values[idx])[0]
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                # (n_samples, n_features, n_classes)
                class_shap_values = arr[0, :, predicted_index]
            elif arr.ndim == 2:
                # (n_samples, n_features)
                class_shap_values = arr[0]
            else:
                class_shap_values = np.ravel(arr)

        feature_impacts = list(zip(self._ORDERED_FEATURES, class_shap_values))
        feature_impacts.sort(key=lambda x: abs(float(x[1])), reverse=True)

        top: Dict[str, Any] = {}
        for feature_name, shap_val in feature_impacts[:3]:
            raw_val = raw_df[feature_name].iloc[0] if feature_name in raw_df.columns else "Unknown"
            # Keep raw_val JSON-serializable
            if isinstance(raw_val, (np.integer,)):
                raw_val = int(raw_val)
            elif isinstance(raw_val, (np.floating,)):
                raw_val = float(raw_val)
            top[feature_name] = {
                "raw_value": raw_val,
                "shap_impact": round(float(shap_val), 4),
            }
        return top