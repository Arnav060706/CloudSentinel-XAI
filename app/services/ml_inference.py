# app/services/ml_inference.py
"""
ParallelMLEngine — Asynchronous Multi-Model Inference & XAI Extraction
======================================================================

Implements the Layer 3 inference block for CloudSentinel-XAI:

  1. Unsupervised Anomaly Detection (Isolation Forest)
     Evaluates the raw telemetry feature vector to produce a baseline
     structural anomaly score. Raw decision_function values [-0.5, 0.5]
     are sigmoid-remapped to a normalized [0, 1] risk probability.

  2. Supervised Threat Phase Classification (XGBoost)
     Classifies the exact MITRE ATT&CK phase (e.g., Privilege Escalation).

  3. SHAP Feature Attribution (TreeSHAP)
     Calculates the marginal contribution of every feature toward the 
     XGBoost prediction. This vector is passed to the XAI Faithfulness 
     Gate to ensure the Llama 3.2 LLM narrative is mathematically grounded.

Concurrency Note
----------------
Machine Learning inference is CPU-bound. To prevent blocking the FastAPI 
ASGI event loop, all model predictions are wrapped in `asyncio.to_thread()` 
and executed concurrently via `asyncio.gather()`.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Attempt to import shap, fail gracefully if not installed in the current environment
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class ParallelMLEngine:
    """
    Handles non-blocking, parallel ML inference.
    
    Expects models to be passed in from the global FastAPI lifespan state
    to prevent reloading binary files from disk on every network request.
    """
    
    # ------------------------------------------------------------------ #
    # The Expected Feature Schema (X Matrix)                             #
    # Must strictly exclude 'threat_category' and 'incident_id' to       #
    # prevent target leakage.                                            #
    # ------------------------------------------------------------------ #
    _ORDERED_FEATURES = [
        "source_cloud",
        "principal_type",
        "principal_created_in_window",
        "is_known_proxy_or_tor",
        "ip_geo_country",
        "ua_family",
        "ua_version",
        "action",
        "status"
    ]

    def __init__(self, state_matrix: dict):
        """
        Extracts loaded models and encoders from the FastAPI global state.
        """
        self.iso_forest = state_matrix.get("iso_forest")
        self.xgboost_model = state_matrix.get("xgboost")
        self.feature_encoder = state_matrix.get("feature_encoder")
        
        if not self.iso_forest or not self.xgboost_model:
            logger.warning("MLEngine initialized without loaded models. Inference will be bypassed.")

        # Initialize TreeSHAP explainer if XGBoost is present
        self.explainer = None
        if SHAP_AVAILABLE and self.xgboost_model:
            self.explainer = shap.TreeExplainer(self.xgboost_model)

    # ------------------------------------------------------------------ #
    # Public Async API                                                   #
    # ------------------------------------------------------------------ #

    async def execute_parallel_inference(self, normalized_event: dict) -> dict:
        """
        Main entry point. Prepares features and runs IF and XGB in parallel.
        Returns the enriched dictionary to be pushed to the Hawkes Risk Engine.
        """
        if not self.iso_forest or not self.xgboost_model:
            # Failsafe: if models aren't loaded, return standard weights
            normalized_event["anomaly_score"] = 0.1
            normalized_event["phase_confidence"] = 0.1
            return normalized_event

        # 1. Prepare and encode the feature matrix (X)
        try:
            x_tensor, raw_df = self._prepare_feature_vector(normalized_event)
        except Exception as e:
            logger.error(f"Feature encoding failure: {e}")
            normalized_event["anomaly_score"] = 0.1
            return normalized_event

        # 2. Execute CPU-bound models in isolated background threads concurrently
        iso_task = asyncio.to_thread(self._run_isolation_forest, x_tensor)
        xgb_task = asyncio.to_thread(self._run_xgboost_and_shap, x_tensor, raw_df)
        
        iso_result, xgb_result = await asyncio.gather(iso_task, xgb_task)

        # 3. Mutate the original event payload with ML context
        normalized_event["anomaly_score"] = iso_result
        
        normalized_event["predicted_phase"] = xgb_result.get("predicted_phase", "Normal")
        normalized_event["phase_confidence"] = xgb_result.get("confidence", 0.0)
        
        # Pass SHAP attributions down the pipeline for the LLM XAI Gate
        normalized_event["shap_attributions"] = xgb_result.get("shap_attributions", {})

        return normalized_event

    # ------------------------------------------------------------------ #
    # Internal Threaded Workers                                          #
    # ------------------------------------------------------------------ #

    def _prepare_feature_vector(self, event: dict) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Isolates permitted features and applies preprocessing (One-Hot / Ordinal).
        """
        # Extract only allowed features (preventing target leakage)
        raw_features = {feat: event.get(feat, "Unknown") for feat in self._ORDERED_FEATURES}
        df = pd.DataFrame([raw_features])
        
        # Production systems require string categorical variables to be mapped to 
        # numeric tensors before hitting tree models. 
        if self.feature_encoder:
            encoded_tensor = self.feature_encoder.transform(df)
        else:
            # Fallback for development if encoder isn't provided:
            # XGBoost can handle 'category' dtypes if configured natively
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype == 'bool':
                    df[col] = df[col].astype('category')
            encoded_tensor = df

        return encoded_tensor, df

    def _run_isolation_forest(self, x_tensor) -> float:
        """
        Executes Isolation Forest and maps the output to a normalized [0,1] score.
        """
        # decision_function returns lower scores (negative) for anomalies.
        # e.g., -0.2 is anomalous, +0.1 is normal.
        raw_score = self.iso_forest.decision_function(x_tensor)[0]
        
        # Remap to [0, 1] where 1.0 is highly anomalous (inverted)
        # Using a clamped linear scale assuming standard IF range of [-0.5, 0.5]
        normalized_score = 0.5 - raw_score
        return round(max(0.0, min(1.0, normalized_score)), 4)

    def _run_xgboost_and_shap(self, x_tensor, raw_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes Phase Classification and extracts TreeSHAP attributions.
        """
        # 1. Predict Class and Probability
        probabilities = self.xgboost_model.predict_proba(x_tensor)[0]
        predicted_index = np.argmax(probabilities)
        confidence = probabilities[predicted_index]
        
        # Map numeric class back to string (Assuming classes were mapped via encoder)
        classes = getattr(self.xgboost_model, "classes_", ["Normal", "PrivilegeEscalation", "DefenseEvasion"])
        predicted_phase = classes[predicted_index] if predicted_index < len(classes) else "Unknown"
        
        # 2. Extract SHAP values for Explainable AI Narrative
        top_shap_features = {}
        if self.explainer:
            # Calculate marginal contributions for this specific prediction
            shap_values = self.explainer.shap_values(x_tensor)
            
            # For multi-class, shap_values is a list of arrays. Grab the array for the predicted class.
            if isinstance(shap_values, list):
                class_shap_values = shap_values[predicted_index][0]
            else:
                class_shap_values = shap_values[0]
                
            # Zip feature names with their SHAP magnitudes
            feature_impacts = list(zip(self._ORDERED_FEATURES, class_shap_values))
            
            # Sort by absolute impact magnitude to find the true drivers of the prediction
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Select top 3 driving features for the LLM prompt
            for feature_name, shap_val in feature_impacts[:3]:
                # Include the raw value of the feature so the LLM can say: 
                # "Triggered because source_cloud was AWS"
                raw_val = raw_df[feature_name].iloc[0]
                top_shap_features[feature_name] = {
                    "raw_value": raw_val,
                    "shap_impact": round(float(shap_val), 4)
                }

        return {
            "predicted_phase": str(predicted_phase),
            "confidence": round(float(confidence), 4),
            "shap_attributions": top_shap_features
        }