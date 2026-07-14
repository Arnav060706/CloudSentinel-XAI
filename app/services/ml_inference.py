"""
ParallelMLEngine — Asynchronous Multi-Model Inference & XAI Extraction
======================================================================

Layer 3 inference: Isolation Forest anomaly score + XGBoost ATT&CK phase
classification + TreeSHAP attributions. CPU-bound work is wrapped in
asyncio.to_thread and run concurrently.

REWRITTEN to match the models actually trained by models/train_isolation_forest.py
and models/train_xgboost.py. The previous version assumed a single shared
9-field raw feature list (_ORDERED_FEATURES) and one generic feature_encoder
serving both models -- neither exists. The two models were deliberately
trained with DIFFERENT feature sets and DIFFERENT fitted label encoders (see
feature_extractor.py's include_labeled_only_features param and its docstring
for why: IF is trained benign-only, so geo_country/is_known_proxy_or_tor/
device_compliant_status/is_internal_ip are zero-variance there; XGBoost is
trained on labeled attack data, where those same columns have real,
confirmed variance -- a single shared encoding could not correctly serve
both). This version builds two separately-encoded feature vectors per event,
one per model, via MLFeatureExtractor reusing each model's own fitted
label_encoders/feature_columns (exactly as models/scoring_utils.py's
featurize_and_align does for offline evaluation).

state_matrix contract (CHANGED): expects "iso_forest_bundle" and
"xgboost_bundle" -- the full dicts pickle.load()'d from models/iso_forest.pkl
/ models/xgboost_classifier.pkl (keys: model, label_encoders, feature_columns,
and for xgboost_bundle also class_names) -- not separate "iso_forest" /
"xgboost" / "feature_encoder" / "class_names" keys. See app/main.py's
_load_ml_artifacts.

ROLLING FEATURES [Phase 5 — FIXED]: MLFeatureExtractor computes several
rolling/velocity features (api_call_count_1m, unique_ips_last_24h,
error_rate_5m, etc.) relative to a user's recent event HISTORY. Previously,
called on a single incoming event with no history, those features took their
"first event ever seen for this user" defaults (train/serve skew). Now, when
stateful_features=True (default for the live engine), an EventHistoryBuffer
(app/services/event_history.py) keeps a bounded trailing per-user history and
each event is featurized WITH that history; the current event's engineered row
is recovered by __eval_row_id__ (never by position). Set stateful_features=False
for batch/offline scoring, which already featurizes whole batches at once.

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
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.parser_normalizer.src.feature_extractor import MLFeatureExtractor
from app.services.event_history import EventHistoryBuffer

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class ParallelMLEngine:
    """Non-blocking, parallel ML inference reading models from state_matrix."""

    # DEPRECATED — kept only so app/services/xai_engine.py's faithfulness
    # deletion test (which still targets the old 9-field raw scheme) doesn't
    # crash on import/attribute access. NOT used by the inference path below
    # anymore; see the module docstring's DISCLOSED LIMITATION and
    # models/README.md's Known gaps for why xai_engine.py's deletion test
    # needs its own follow-up fix (it needs the full ENGINEERED feature
    # vector used for the original prediction, not just raw event fields).
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

    def __init__(self, state_matrix: dict, stateful_features: bool = True):
        # Phase 5: when stateful_features is True (default for the live engine),
        # each event is featurized WITH the user's recent trailing history so
        # rolling/velocity features (api_call_count_1m, unique_ips_last_24h, ...)
        # reflect real activity instead of first-event defaults. Set False for
        # batch/offline scoring, which already featurizes whole batches at once
        # (models/scoring_utils.featurize_and_align) and must be unaffected.
        self.stateful_features = stateful_features
        self.history = EventHistoryBuffer() if stateful_features else None

        iso_bundle = state_matrix.get("iso_forest_bundle")
        xgb_bundle = state_matrix.get("xgboost_bundle")

        self.iso_forest = iso_bundle.get("model") if iso_bundle else None
        self.iso_label_encoders = iso_bundle.get("label_encoders") if iso_bundle else None
        self.iso_feature_columns = iso_bundle.get("feature_columns") if iso_bundle else None

        self.xgboost_model = xgb_bundle.get("model") if xgb_bundle else None
        if self.xgboost_model is not None:
            # models/train_xgboost.py trains on GPU (device="cuda") by
            # default. Confirmed empirically: a GPU-resident booster crashes
            # the process at a native level (no Python traceback, not a
            # catchable exception) when combined with shap.TreeExplainer --
            # reproduced directly, isolated to this exact combination.
            # Switching to CPU for inference is free (a single-event predict
            # call has no meaningful GPU benefit anyway, unlike training) and
            # eliminates the crash entirely -- verified after this change.
            self.xgboost_model.set_params(device="cpu")
        self.xgb_label_encoders = xgb_bundle.get("label_encoders") if xgb_bundle else None
        self.xgb_feature_columns = xgb_bundle.get("feature_columns") if xgb_bundle else None
        self.class_names = xgb_bundle.get("class_names") if xgb_bundle else None

        # Phase 4: the risk path consumes XGBoost's 1 - P(Normal) (the
        # B2-winning signal, validated offline in evaluate_full_pipeline.py), so
        # cache the index of the "Normal" class once. None => fall back to the
        # Isolation Forest score (logged once) rather than silently mis-scoring.
        self.xgb_normal_index = (
            self.class_names.index("Normal")
            if self.class_names and "Normal" in self.class_names else None
        )
        self._warned_xgb_fallback = False

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
        event["if_anomaly_score"] = 0.1  # keep the field present on every path
        event["predicted_phase"] = "Normal"
        event["predicted_phase_index"] = 0
        event["phase_confidence"] = 0.1
        event["shap_attributions"] = {}
        return event

    async def execute_parallel_inference(self, normalized_event: dict) -> dict:
        if not self._models_ready:
            return self._bypass(normalized_event)

        # 1. Prepare features -- TWO separately-encoded tensors, one per
        #    model (see module docstring for why one shared tensor can't
        #    correctly serve both). Phase 5: in stateful mode, featurize the
        #    event WITH its user's recent trailing history so rolling/velocity
        #    features are real, not first-event defaults.
        try:
            history_batch = None
            if self.history is not None:
                history_batch = self.history.add_and_snapshot(normalized_event)
            x_iso, x_xgb, raw_df = self._prepare_feature_vectors(normalized_event, history_batch)
        except Exception as e:
            logger.error("Feature encoding failure: %s", e)
            return self._bypass(normalized_event)

        # 2. Run models concurrently in threads
        try:
            iso_task = asyncio.to_thread(self._run_isolation_forest, x_iso)
            xgb_task = asyncio.to_thread(self._run_xgboost_and_shap, x_xgb, raw_df)
            iso_result, xgb_result = await asyncio.gather(iso_task, xgb_task)
        except Exception as e:
            logger.error("Inference failure: %s", e)
            return self._bypass(normalized_event)

        # 3. Enrich the event.
        # Phase 4: feed XGBoost's stronger per-event signal into the risk path.
        # risk_engine.py reads "anomaly_score" -> set it to XGBoost's
        # 1 - P(Normal) (its own "how likely is this ANY attack" estimate, the
        # B2-winning row offline: ROC-AUC 0.68 vs the IF's 0.53). The Isolation
        # Forest score is preserved as "if_anomaly_score" -- NEVER discarded;
        # it's needed for ablations and as the fallback below. Both are in
        # [0,1]; because Phase 2 recalibrates the alert threshold FROM DATA
        # against this same 1-P(Normal) scale, the swap needs no other change.
        normalized_event["if_anomaly_score"] = iso_result
        xgb_attack_score = xgb_result.get("attack_score")
        if xgb_attack_score is None:
            # XGBoost bundle lacked a usable "Normal" class -> keep the risk
            # path alive on the IF signal instead, warning exactly once.
            if not self._warned_xgb_fallback:
                logger.warning(
                    "XGBoost attack score unavailable (no 'Normal' class in "
                    "bundle) — falling back to Isolation Forest anomaly_score "
                    "for the risk path. This warning is logged once."
                )
                self._warned_xgb_fallback = True
            normalized_event["anomaly_score"] = iso_result
        else:
            normalized_event["anomaly_score"] = round(float(xgb_attack_score), 4)
        normalized_event["predicted_phase"] = xgb_result.get("predicted_phase", "Normal")
        normalized_event["predicted_phase_index"] = xgb_result.get("predicted_phase_index", 0)
        normalized_event["phase_confidence"] = xgb_result.get("confidence", 0.0)
        normalized_event["shap_attributions"] = xgb_result.get("shap_attributions", {})
        # Phase 6: carry the EXACT engineered XGBoost feature row used for this
        # prediction (including live rolling-window state), so xai_engine's
        # faithfulness deletion test operates in engineered space on the vector
        # the model actually saw -- not raw event fields reassembled after the
        # fact (which wouldn't reproduce hour_of_day/api_call_count_1m/etc.).
        try:
            first = x_xgb.iloc[0]
            normalized_event["xgb_feature_row"] = {
                c: float(first[c]) for c in self.xgb_feature_columns
            }
        except Exception as e:
            logger.error("Could not attach xgb_feature_row: %s", e)
        return normalized_event

    # ------------------------------------------------------------------ #
    # Internal threaded workers                                          #
    # ------------------------------------------------------------------ #

    def _encode_for(self, batch_with_ids: list, current_row_id: int, label_encoders,
                     feature_columns, include_labeled_only_features: bool) -> pd.DataFrame:
        """Featurize a user's trailing history batch, reusing the model's own
        fitted label encoders, and return ONLY the current event's engineered
        row. The current row is recovered by its unique __eval_row_id__, NEVER by
        position -- extract_features() sorts by timestamp internally, so the
        current event is not reliably last (Global rule 1 /
        models/scoring_utils.featurize_and_align, replicated here to avoid the
        live path depending on the offline models/ package)."""
        extractor = MLFeatureExtractor()
        extractor.label_encoders = label_encoders
        X, _ = extractor.extract_features(
            batch_with_ids, is_training=False,
            include_labeled_only_features=include_labeled_only_features,
        )
        X = X[X["__eval_row_id__"] == current_row_id]
        # reindex drops __eval_row_id__ (and any stray enrichment columns) and
        # orders exactly to the model's trained feature set.
        return X.reindex(columns=feature_columns, fill_value=0)

    def _prepare_feature_vectors(self, event: dict, history_batch=None
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Stateless (batch/offline or first-ever event) collapses to [event];
        # stateful passes the user's trailing history with `event` LAST.
        batch = history_batch if history_batch is not None else [event]
        batch_with_ids = [dict(e, __eval_row_id__=i) for i, e in enumerate(batch)]
        current_row_id = len(batch_with_ids) - 1  # the just-added current event

        x_iso = self._encode_for(
            batch_with_ids, current_row_id, self.iso_label_encoders,
            self.iso_feature_columns, include_labeled_only_features=False,
        )
        x_xgb = self._encode_for(
            batch_with_ids, current_row_id, self.xgb_label_encoders,
            self.xgb_feature_columns, include_labeled_only_features=True,
        )
        # Raw (pre-engineering) CURRENT event as a single-row DataFrame, for
        # SHAP's human-readable "raw_value" reporting. Engineered-only feature
        # names (hour_of_day, api_call_count_1m, etc.) aren't present here and
        # fall back to "Unknown" in _extract_shap -- an honest degradation, not
        # a crash, consistent with the rest of this file's defensive style.
        raw_df = pd.DataFrame([event])
        return x_iso, x_xgb, raw_df

    def _run_isolation_forest(self, x_tensor) -> float:
        raw_score = self.iso_forest.decision_function(x_tensor)[0]
        # Lower (more negative) => more anomalous. Remap to [0,1] where 1 is
        # highly anomalous. decision_function is not guaranteed to sit in
        # [-0.5, 0.5]; clamp defensively. risk_engine.py's anomaly clamp
        # expects exactly this convention -- see models/evaluate_full_pipeline.py's
        # module docstring for why the plain -decision_function() convention
        # (used elsewhere for pure ranking purposes) would silently corrupt
        # this specific downstream consumer.
        normalized_score = 0.5 - raw_score
        return round(max(0.0, min(1.0, normalized_score)), 4)

    def _run_xgboost_and_shap(self, x_tensor, raw_df: pd.DataFrame) -> Dict[str, Any]:
        probabilities = np.asarray(self.xgboost_model.predict_proba(x_tensor)[0])
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        predicted_phase = self._phase_name(predicted_index)

        # Phase 4: continuous "how likely is this ANY attack" signal for the
        # risk path = 1 - P(Normal). None if the bundle has no "Normal" class,
        # so execute_parallel_inference can fall back to the IF score.
        attack_score = None
        if self.xgb_normal_index is not None and self.xgb_normal_index < len(probabilities):
            attack_score = float(1.0 - probabilities[self.xgb_normal_index])

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
            "attack_score": attack_score,
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

        # Zipped against the XGBoost model's ACTUAL trained feature columns
        # (self.xgb_feature_columns), not the old fixed _ORDERED_FEATURES --
        # x_tensor's columns are exactly xgb_feature_columns (see
        # _prepare_feature_vectors -> _encode_for's reindex).
        feature_impacts = list(zip(self.xgb_feature_columns, class_shap_values))
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
