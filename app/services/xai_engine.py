# app/services/xai_engine.py
import asyncio
import logging
import pandas as pd
from typing import Dict, Any, Tuple
from ollama import AsyncClient, ResponseError

logger = logging.getLogger(__name__)

class FaithfulnessGatedXAI:
    def __init__(self, state_matrix: dict, ollama_host: str = 'http://localhost:11434'):
        """
        Initializes the XAI engine, extracting the XGBoost model to perform
        the mathematical deletion test (Faithfulness Gate) prior to LLM execution.
        """
        self.xgboost_model = state_matrix.get("xgboost")
        self.feature_encoder = state_matrix.get("feature_encoder")
        
        # Utilize the non-blocking AsyncClient to protect the FastAPI event loop
        self.llm_client = AsyncClient(host=ollama_host)
        
        # Delta threshold: Confidence must drop by at least 15% when top features 
        # are removed for the explanation to be considered mathematically "faithful".
        self.faithfulness_delta = 0.15 

    async def generate_forensic_narrative(self, log_data: dict, risk_state: dict) -> Tuple[bool, str]:
        """
        Main Layer 5 Entrypoint.
        Executes the Faithfulness Gate. If passed, queries Llama 3.2 asynchronously.
        Returns: (Passed_Gate_Bool, Narrative_String)
        """
        shap_attributions = log_data.get("shap_attributions", {})
        original_confidence = log_data.get("phase_confidence", 0.0)
        predicted_phase = log_data.get("predicted_phase", "Unknown")
        
        # Extract the original class index; defaults to 0 if missing to prevent crashes
        # This should ideally be passed from the upstream ml_inference dictionary (subject to change)
        predicted_phase_index = log_data.get("predicted_phase_index", 0) 
        
        # Step 1: Execute the Faithfulness Deletion Gate (CPU-Bound, sent to thread)
        is_faithful = await asyncio.to_thread(
            self._run_deletion_test, 
            log_data, 
            shap_attributions, 
            original_confidence,
            predicted_phase_index
        )
        
        if not is_faithful:
            logger.warning(f"XAI Faithfulness Gate FAILED for entity: {log_data.get('principal', 'Unknown')}")
            return False, "Low-confidence attribution: SHAP deletion test failed. Flagged for manual analyst review."

        # Step 2: Gate Passed: Construct the rigorous prompting context
        system_instruction = (
            "You are an expert cloud security AI agent embedded within a Tier-3 SOC. "
            "Your task is to translate complex multi-cloud risk telemetry and validated "
            "mathematical SHAP attributions into an actionable triage narrative. "
            "Limit your response to exactly two sentences. Do not hallucinate external context."
        )
        
        # Incorporate the Hawkes dominant signal directly to guide the LLM's reasoning
        user_prompt = f"""
        [CRITICAL MULTI-CLOUD SECURITY ALERT]
        Target Entity: {log_data.get('principal', 'Unknown')}
        Cloud Provider: {log_data.get('source_cloud', 'Unknown')}
        Detected Attack Phase: {predicted_phase} (Confidence: {original_confidence * 100:.1f}%)
        
        Hawkes Risk Engine Dominant Signal: {risk_state.get('dominant_signal', 'Unknown')}
        Cross-Cloud Span: {risk_state.get('cloud_span_count', 1)} providers
        
        Validated SHAP Feature Drivers:
        {shap_attributions}
        
        Provide a 2-sentence tactical breakdown explaining what happened and why the AI flagged it based strictly on these features.
        """

        # Step 3: Asynchronous LLM Execution; AyncClient is handling the event loop without blocking FastAPI
        try:
            response = await self.llm_client.chat(
                model='llama3.2',
                messages=[
                    {'role': 'system', 'content': system_instruction},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': 0.1,  # Ultra-low temp for deterministic reporting
                    'num_predict': 150
                }
            )
            # Safely navigate the response dictionary
            narrative = response.get('message', {}).get('content', '').strip()
            return True, narrative
            
        except ResponseError as e:
            logger.error(f"Ollama Inference Error: {e}")
            return True, "Automated narrative generation failed due to inference timeout."
        except Exception as e:
            logger.error(f"Unexpected XAI exception: {e}")
            return True, "Automated narrative generation encountered an unexpected error."

    def _run_deletion_test(self, log_data: dict, shap_attributions: dict, original_confidence: float, original_class_index: int) -> bool:
        """
        The Mathematical Faithfulness Gate (Patent/Paper Claim).
        Zeroes out the top SHAP features, re-encodes, and re-runs XGBoost.
        If the prediction confidence does not drop by the delta threshold, the
        attribution is spurious/unfaithful.
        """
        if not self.xgboost_model or not shap_attributions:
            return False
            
        # Pre-define exclusion keys as a set for faster O(1) lookups during dictionary comprehension
        exclude_keys = {"anomaly_score", "predicted_phase", "predicted_phase_index", "phase_confidence", "shap_attributions"}
        
        # Create a deep copy of the original feature state to perturb
        perturbed_features = {k: v for k, v in log_data.items() if k not in exclude_keys}
        
        # "Delete" the top SHAP features by setting them to a baseline/unknown state
        for feature_name in shap_attributions.keys():
            if feature_name in perturbed_features:
                val = perturbed_features[feature_name]
                # Maintain type integrity to prevent XGBoost tensor errors
                if isinstance(val, str):
                    perturbed_features[feature_name] = "Unknown"
                elif isinstance(val, bool):
                    perturbed_features[feature_name] = False
                else:
                    perturbed_features[feature_name] = 0.0

        # Re-encode the perturbed vector
        df = pd.DataFrame([perturbed_features])
        
        if self.feature_encoder:
            try:
                x_tensor = self.feature_encoder.transform(df)
            except Exception as e:
                logger.error(f"Feature encoding failed during deletion test: {e}")
                return False
        else:
            # Batch type conversion using pandas vectorization instead of a Python for-loop for efficiency
            obj_cols = df.select_dtypes(include=['object', 'bool']).columns
            if not obj_cols.empty:
                df[obj_cols] = df[obj_cols].astype('category')
            x_tensor = df
            
        # Re-run XGBoost inference
        try:
            probabilities = self.xgboost_model.predict_proba(x_tensor)[0]
        except Exception as e:
            logger.error(f"XGBoost inference failed during deletion test: {e}")
            return False
            
        # Calculate confidence drop on the original predicted class index
        if original_class_index < len(probabilities):
            new_confidence = probabilities[original_class_index]
        else:
            logger.error(f"Class index {original_class_index} out of bounds for probability array.")
            return False
        
        confidence_drop = original_confidence - new_confidence
        
        # If the drop is greater than our threshold, the features were truly load-bearing
        return confidence_drop >= self.faithfulness_delta
    
    # Threshold value is a placeholder for now; subject to change