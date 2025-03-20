import torch
import transformers
from openai import OpenAI
from opensae.transformer_with_sae import TransformerWithSae, InterventionConfig
from opensae import OpenSae

############################################
#           OpenAI API Configuration
############################################
client = OpenAI(
    base_url="https://your-api-url.com/v1",  # Masked URL
    api_key="your-api-key"  # Masked API key
)

############################################
#       GPT Comparison Function (judge part)
############################################

def generate_judge_prompt(text_a, text_b, feature):
    """
    Generate prompt to compare text_a and text_b on a specified feature.
    Returns a response asking for "Text A" or "Text B" with a brief explanation.
    """
    return f"""
Please compare the following two texts based on **{feature}**:

- **Text A**: "{text_a}"
- **Text B**: "{text_b}"

Answer the following question:
Which text demonstrates a stronger presence of **{feature}**? (Please answer only "Text A" or "Text B".)
"""

def compare_texts_gpt(text_a, text_b, feature="syntactic complexity"):
    """
    Call GPT to compare text_a and text_b on a specified linguistic feature.
    Returns "A" or "B" indicating the stronger text.
    """
    prompt = generate_judge_prompt(text_a, text_b, feature)
    response = client.chat.completions.create(
        model="gpt-4",  # Masked model name
        messages=[{"role": "user", "content": prompt}],
        temperature=0.01,
        max_tokens=10,
    )
    content = response.choices[0].message.content.strip()

    # Find the first answer (either "Text A" or "Text B")
    first_line = content.split('\n')[0].lower()
    if "text a" in first_line:
        return "A"
    elif "text b" in first_line:
        return "B"
    else:
        # Search in full content if not in the first line
        if "text a" in content.lower():
            return "A"
        elif "text b" in content.lower():
            return "B"
        else:
            return "Unknown"

############################################
#       Multi-experiment + GPT judge Main Process
############################################
def run_experiments_and_judge(
    input_text,
    model,
    tokenizer,
    full_intervention_indices,  # List of basis vectors
    feature="syntactic complexity",
    num_generations=25
):
    """
    Run 5 experiments for the input text with varying numbers of basis vectors.
    Each experiment performs 3 comparisons (Ablation vs. Invariant, Enhancement vs. Invariant, Enhancement vs. Ablation).
    Returns a dictionary with the frequency stats of each comparison.
    """
    experiments_results = {}
    
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Run experiments for 1 to 5 basis vectors
    for num_basis in range(1, 6):
        print(f"\n=== Experiment with first {num_basis} basis vector(s) ===")
        intervention_indices = full_intervention_indices[:num_basis]
        print(f"Using intervention indices: {intervention_indices}")
        
        # Initialize counters
        count_enh_better_inv = 0  # Enhancement > Invariant
        count_abl_worse_inv = 0   # Ablation < Invariant
        count_enh_better_abl = 0  # Enhancement > Ablation
        
        for gen_i in range(1, num_generations + 1):
            print(f"\n----- Generation {gen_i}/{num_generations} -----")
            # --- 1. Ablation: set mode, value 0
            abl_config = InterventionConfig(
                intervention=True,
                intervention_mode="set",
                intervention_indices=intervention_indices,
                intervention_value=0.0,
                prompt_only=False,
            )
            model.update_intervention_config(abl_config)
            y_abl = model.generate(**inputs, max_new_tokens=100, temperature=1.0)
            text_abl = tokenizer.decode(y_abl[0], skip_special_tokens=True)
            
            # --- 2. Enhancement: set mode, value 10
            enh_config = InterventionConfig(
                intervention=True,
                intervention_mode="set",
                intervention_indices=intervention_indices,
                intervention_value=10.0,
                prompt_only=False,
            )
            model.update_intervention_config(enh_config)
            y_enh = model.generate(**inputs, max_new_tokens=100, temperature=1.0)
            text_enh = tokenizer.decode(y_enh[0], skip_special_tokens=True)
            
            # --- 3. Invariant: multiply mode, value 1
            inv_config = InterventionConfig(
                intervention=True,
                intervention_mode="multiply",
                intervention_indices=intervention_indices,
                intervention_value=1.0,
                prompt_only=False,
            )
            model.update_intervention_config(inv_config)
            y_inv = model.generate(**inputs, max_new_tokens=100, temperature=1.0)
            text_inv = tokenizer.decode(y_inv[0], skip_special_tokens=True)
            
            # --- Compare three pairs ---
            result0 = compare_texts_gpt(text_inv, text_abl, feature)
            if result0 == "A":
                count_abl_worse_inv += 1
            
            result1 = compare_texts_gpt(text_enh, text_inv, feature)
            if result1 == "A":
                count_enh_better_inv += 1
            
            result2 = compare_texts_gpt(text_abl, text_enh, feature)
            if result2 == "B":
                count_enh_better_abl += 1
        
        # Store experiment results
        experiments_results[num_basis] = {
            "Enhanced_Causal_Effect": count_enh_better_inv / num_generations,
            "Ablation_Causal_Effect": count_abl_worse_inv / num_generations,
            "Overlay_Causal_Effect": count_enh_better_abl / num_generations
        }
        
        print(f"\n--- Results for {num_basis} basis vector(s) ---")
        print(f"Enhanced Causal Effect (Enhancement > Invariant): {experiments_results[num_basis]['Enhanced_Causal_Effect']:.2f}")
        print(f"Ablation Causal Effect (Ablation < Invariant): {experiments_results[num_basis]['Ablation_Causal_Effect']:.2f}")
        print(f"Overlay Causal Effect  (Enhancement > Ablation): {experiments_results[num_basis]['Overlay_Causal_Effect']:.2f}")
    
    return experiments_results

def main():
    # ========== 1. Configuration ==========
    device = "cuda:3"  # Adjust based on GPU or "cpu"
    input_text_path = "/path/to/input/polite.txt"  # Masked path
    experiment_name = "polite"
    
    # Define basis vector list (fixed order)
    intervention_indices = [149519, 230365, 154928, 138294]

    # Read input text
    with open(input_text_path, "r", encoding="utf-8") as file:
        input_text = file.read()
    
    # ========== 2. Load Model and Tokenizer ==========
    sae = OpenSae.from_pretrained("/path/to/sae_checkpoint")  # Masked path
    model = TransformerWithSae("/path/to/llama_model", sae, device)  # Masked path
    tokenizer = transformers.AutoTokenizer.from_pretrained("/path/to/llama_tokenizer")  # Masked path
    
    # ========== 3. Run Experiments + GPT Judge Comparison ==========
    feature_to_compare = ("Politeness Significance, defined as follows: "
                         "Politeness Significance refers to the degree to which politeness strategies are salient, effective, and contextually integrated in communication. "
                         "It encompasses their frequency, pragmatic depth, and social impact in shaping interpersonal rapport, mitigating face threats, and reinforcing cooperative intent.")
    
    results = run_experiments_and_judge(
        input_text=input_text,
        model=model,
        tokenizer=tokenizer,
        full_intervention_indices=intervention_indices,
        feature=feature_to_compare,
        num_generations=25
    )
    
    print("\n=== Final Experiment Results ===")
    for num_basis, result in results.items():
        print(f"\nExperiment with {num_basis} basis vector(s):")
        for k, v in result.items():
            print(f"{k}: {v:.2%}")

if __name__ == "__main__":
    main()
