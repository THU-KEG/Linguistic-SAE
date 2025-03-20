import torch
import transformers
from openai import OpenAI

from opensae.transformer_with_sae import TransformerWithSae, InterventionConfig
from opensae import OpenSae

############################################
#          OpenAI API Configuration
############################################
client = OpenAI(
    base_url="https://your-api-url.com/v1",  # Masked API URL
    api_key="your-api-key"  # Masked API key
)

############################################
#           GPT Comparison Function
############################################

def generate_judge_prompt(text_a, text_b, feature):
    return f"""
Please compare the following two texts based on **{feature}**:

- **Text A**: "{text_a}"
- **Text B**: "{text_b}"

Which text shows a stronger presence of **{feature}**? ("Text A" or "Text B")
"""

def compare_texts_gpt(text_a, text_b, feature="syntactic complexity"):
    prompt = generate_judge_prompt(text_a, text_b, feature)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.01,
        max_tokens=10,
    )
    content = response.choices[0].message.content.strip()
    first_line = content.split('\n')[0].lower()
    if "text a" in first_line:
        return "A"
    elif "text b" in first_line:
        return "B"
    else:
        if "text a" in content.lower():
            return "A"
        elif "text b" in content.lower():
            return "B"
        else:
            return "Unknown"

############################################
#    Five Intervention Modes Comparison
############################################

def run_five_mode_intervention_and_judge(
    input_text,
    model,
    tokenizer,
    intervention_indices,
    feature="the salience of metaphor",
    num_generations=10
):
    """
    Perform generation experiments with five intervention modes and compare the results.
    Modes: set_0, set_10, multiply_0, multiply_1, add_10
    Compare outputs using GPT judge.
    """
    # Define intervention modes
    modes = {
        "set_0": InterventionConfig(intervention=True, intervention_mode="set", intervention_indices=intervention_indices, intervention_value=0.0),
        "set_10": InterventionConfig(intervention=True, intervention_mode="set", intervention_indices=intervention_indices, intervention_value=10.0),
        "multiply_0": InterventionConfig(intervention=True, intervention_mode="multiply", intervention_indices=intervention_indices, intervention_value=0.0),
        "multiply_1": InterventionConfig(intervention=True, intervention_mode="multiply", intervention_indices=intervention_indices, intervention_value=1.0),
        "add_10": InterventionConfig(intervention=True, intervention_mode="add", intervention_indices=intervention_indices, intervention_value=10.0),
    }
    
    # Initialize win counts for each pair
    win_counts = {m: {n: 0 for n in modes if n != m} for m in modes}
    mode_names = list(modes.keys())
    total_counts = {(mode_names[i], mode_names[j]): 0 for i in range(len(mode_names)) for j in range(i + 1, len(mode_names))}
    
    # Prepare input for model
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    for gen_i in range(1, num_generations + 1):
        print(f"===== Generation {gen_i} / {num_generations} =====")
        texts = {}
        
        # Generate texts for each mode
        for mode_name, config in modes.items():
            model.update_intervention_config(config)
            y = model.generate(**inputs, max_new_tokens=50, temperature=1.0)
            text = tokenizer.decode(y[0], skip_special_tokens=True)
            texts[mode_name] = text
        
        # Compare all mode pairs
        for i in range(len(mode_names)):
            for j in range(i + 1, len(mode_names)):
                mode_i = mode_names[i]
                mode_j = mode_names[j]
                print(f"Comparing {mode_i} vs {mode_j}")
                winner = compare_texts_gpt(texts[mode_i], texts[mode_j], feature)
                if winner == "A":
                    win_counts[mode_i][mode_j] += 1
                elif winner == "B":
                    win_counts[mode_j][mode_i] += 1
                total_counts[(mode_i, mode_j)] += 1
    
    # Calculate win frequencies
    frequencies = {}
    for (mode_i, mode_j), total in total_counts.items():
        freq_i = win_counts[mode_i][mode_j] / total
        freq_j = win_counts[mode_j][mode_i] / total
        frequencies[(mode_i, mode_j)] = {mode_i: freq_i, mode_j: freq_j}
        print(f"Pair {mode_i} vs {mode_j}: {mode_i} wins {freq_i:.2f}, {mode_j} wins {freq_j:.2f}")
    
    return frequencies

def main():
    # ========== 1. Configuration ==========
    device = "cuda:0"  # Set to GPU or "cpu"
    input_text_path = "/path/to/metaphor.txt"  # Masked input path
    intervention_indices = [75327, 198568, 250513, 230379, 253776]
    
    with open(input_text_path, "r", encoding="utf-8") as file:
        input_text = file.read()
    
    # ========== 2. Load Model and Tokenizer ==========
    sae = OpenSae.from_pretrained("/path/to/sae_checkpoint")  # Masked path
    model = TransformerWithSae("/path/to/llama_model", sae, device)  # Masked path
    tokenizer = transformers.AutoTokenizer.from_pretrained("/path/to/llama_tokenizer")  # Masked path
    
    # ========== 3. Run Intervention Experiment and GPT Judging ==========
    feature_to_compare = "Metaphor Activation, defined as the strength, clarity, and integration of metaphorical expressions."
    frequencies = run_five_mode_intervention_and_judge(
        input_text=input_text,
        model=model,
        tokenizer=tokenizer,
        intervention_indices=intervention_indices,
        feature=feature_to_compare,
        num_generations=25
    )
    
    # ========== 4. Output Results ==========
    print("\n=== Summary ===")
    for (mode_i, mode_j), freq in frequencies.items():
        print(f"Pair {mode_i} vs {mode_j}: {mode_i} wins {freq[mode_i]:.2f}, {mode_j} wins {freq[mode_j]:.2f}")

if __name__ == "__main__":
    main()
