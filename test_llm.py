import ollama

def generate_triage_narrative(model_insights):
    # Constructing a structured prompt for Layer 5 of your project
    prompt = f"""
    You are a Tier-3 SOC Analyst central aggregator. 
    Analyze the following feature weights and log telemetry data, then output a concise, 
    exactly 2-sentence threat narrative explaining why this alert happened.
    
    Data: {model_insights}
    """
    
    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    return response['message']['content']

# Test data mimicking what your SHAP engine will output later
mock_shap_data = {
    "cloud": "AWS",
    "event": "UpdateAssumeRolePolicy",
    "top_shap_features": {"time_of_day_deviation": 0.84, "unauthorized_ip": 0.71},
    "risk_score": 89
}

print("🤖 Querying local LLM aggregator...")
narrative = generate_triage_narrative(mock_shap_data)
print(f"\n📝 Generated Triage Narrative:\n{narrative}")