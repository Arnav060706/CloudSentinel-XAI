import ollama
import json

def generate_soc_narrative(cloud_provider, event_name, risk_score, shap_features):
    """
    Ingests structural telemetry and SHAP importance vectors, 
    and queries local Llama 3.2 for a concise Tier-1 SOC triage narrative.
    """
    
    # Constructing a strict, professional system prompt to prevent LLM hallucination
    system_instruction = (
        "You are an expert cloud security AI agent embedded within a Tier-3 SOC. "
        "Your task is to translate complex machine learning model outputs (SHAP values) "
        "and multi-cloud logs into an actionable, highly professional triage narrative. "
        "Limit your response to exactly two sentences. Focus strictly on the data provided."
    )
    
    user_prompt = f"""
    [CRITICAL SECURITY ALERT ENGINE TRIGGERED]
    Cloud Provider: {cloud_provider}
    Interceptors triggered: Parallel Risk Engine (Isolation Forest & XGBoost)
    Event Action: {event_name}
    Calculated Risk Score: {risk_score}/100
    
    Top SHAP Feature Explanations (Feature Weights indicating anomaly drivers):
    {json.dumps(shap_features, indent=2)}
    
    Provide the 2-sentence tactical breakdown explaining what happened and why the AI flagged it.
    """
    
    try:
        response = ollama.chat(
            model='llama3.2',
            messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': user_prompt}
            ],
            options={
                'temperature': 0.2, # Low temperature ensures deterministic, non-creative security reporting
                'num_predict': 150   # Hard limit on token length to save local CPU cycles
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"LLM Generation Error: {str(e)}"

#if __name__ == "__main__":
    # Test execution block mimicking an AWS credential compromise
    # print("Testing Local Llama 3.2 Aggregator Framework...")
    
    # mock_shap = {
    #     "temporal_deviation_hours": 0.89,  # High weight: Action happened at an unusual time
    #     "untrusted_source_ip_range": 0.76, # High weight: Location anomaly
    #     "api_call_velocity": 0.12          # Low weight: Speed was normal (Low-and-Slow evasion)
    # }
    
    # narrative = generate_soc_narrative(
    #     cloud_provider="AWS", 
    #     event_name="UpdateAssumeRolePolicy", 
    #     risk_score=87.5, 
    #     shap_features=mock_shap
    # )
    
    # print("\nOutput Triage Narrative:")
    # print("-" * 60)
    # print(narrative)
    # print("-" * 60)