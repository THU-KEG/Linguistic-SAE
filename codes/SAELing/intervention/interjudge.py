import torch
import transformers
from openai import OpenAI

from opensae.transformer_with_sae import TransformerWithSae, InterventionConfig
from opensae import OpenSae

############################################
#           OpenAI API Configuration
############################################
client = OpenAI(
    base_url="https://example.com/v1",  # Masked API URL
    api_key="sk-XXXXXXXXXXXXXXXXXXXXXXXXXX"  # Masked API Key
)

############################################
#       GPT Comparison Function (judge part)
############################################

def generate_judge_prompt(text_a, text_b, feature):
    """
    Generate a prompt for GPT to compare two texts based on a specific feature.
    Returns the answer "Text A" or "Text B" with a brief explanation.
    """
    return f"""
Please compare the following two texts based on **{feature}** (e.g., syntactic complexity, lexical richness, semantic clarity, sentiment intensity):

- **Text A**: "{text_a}"
- **Text B**: "{text_b}"

Answer the following question:
Which text demonstrates a stronger presence of **{feature}**? (Please answer only "Text A" or "Text B".)
"""

def compare_texts_gpt(text_a, text_b, feature="syntactic complexity"):
    """
    Use GPT to compare the two texts based on a linguistic feature.
    Returns "A" or "B" indicating the stronger text.
    """
    prompt = generate_judge_prompt(text_a, text_b, feature)
    response = client.chat.completions.create(
        model="gpt-4",  # Masked model version
        messages=[{"role": "user", "content": prompt}],
        temperature=0.01,
        max_tokens=10,
    )

    content = response.choices[0].message.content.strip()
    first_line = content.split('\n')[0].lower()  # Convert to lowercase for matching
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
#       Intervention Experiment + Text Comparison Main Process
############################################

def run_intervention_and_judge(
    input_text,
    model,
    tokenizer,
    intervention_indices,
    feature="syntactic complexity",
    num_generations=10
):
    """
    Perform multiple intervention experiments (Ablation, Enhancement, Invariant) on the input text.
    Compare the results using GPT and calculate the causal effects.
    """
    count_enh_better_inv = 0  # Times Enhancement > Invariant
    count_abl_worse_inv = 0   # Times Ablation < Invariant
    count_enh_better_abl = 0  # Times Enhancement > Ablation

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    for gen_i in range(1, num_generations + 1):
        print(f"===== Generation {gen_i} / {num_generations} =====")

        # Ablation: set value = 0
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

        # Enhancement: set value = 10
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

        # Invariant: multiply value = 1
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

        # Compare the three texts pairwise (order swapped)
        winner_inv_abl = compare_texts_gpt(text_inv, text_abl, feature)
        if winner_inv_abl == "A":
            count_abl_worse_inv += 1

        winner_inv_enh = compare_texts_gpt(text_enh, text_inv, feature)
        if winner_inv_enh == "A":
            count_enh_better_inv += 1

        winner_abl_enh = compare_texts_gpt(text_abl, text_enh, feature)
        if winner_abl_enh == "B":
            count_enh_better_abl += 1

    # Calculate causal effects
    total = num_generations
    enhanced_causal_effect = count_enh_better_inv / total
    ablation_causal_effect = count_abl_worse_inv / total
    overlay_causal_effect = count_enh_better_abl / total

    print("===== Final Statistics =====")
    print(f"Enhanced Causal Effect (Enhancement > Invariant): {enhanced_causal_effect:.2f}")
    print(f"Ablation Causal Effect (Ablation < Invariant): {ablation_causal_effect:.2f}")
    print(f"Overlay Causal Effect  (Enhancement > Ablation): {overlay_causal_effect:.2f}")

    return {
        "Enhanced_Causal_Effect": enhanced_causal_effect,
        "Ablation_Causal_Effect": ablation_causal_effect,
        "Overlay_Causal_Effect": overlay_causal_effect
    }

def main():
    # ========== 1. Basic Configuration ==========
    device = "cuda:0"  # Set GPU or "cpu" based on availability
    input_text_path = "/path/to/input.txt"  # Masked input file path
    experiment_name = "inverse"
    intervention_indices = [17802]

    # Read input text
    with open(input_text_path, "r", encoding="utf-8") as file:
        input_text = file.read()

    # ========== 2. Load Model and Tokenizer =========
    sae = OpenSae.from_pretrained("/path/to/sae_model")  # Masked SAE model path
    model = TransformerWithSae("/path/to/llama_model", sae, device)  # Masked LLAMA model path
    tokenizer = transformers.AutoTokenizer.from_pretrained("/path/to/llama_model")  # Masked tokenizer path

    # ========== 3. Run Intervention Experiment + GPT judge ==========
    feature_to_compare = ("Inverse Verb Fronting Significance, defined as follows: "
                          "Verb Fronting Significance refers to the degree to which verb fronting—where the verb is positioned before the subject or at the beginning of a clause—is used effectively and contextually appropriately in a text.")

    results = run_intervention_and_judge(
        input_text=input_text,
        model=model,
        tokenizer=tokenizer,
        intervention_indices=intervention_indices,
        feature=feature_to_compare,
        num_generations=25  # Perform 25 generations
    )

    # ========== 4. Output Results ==========
    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"{k}: {v:.2f}")

if __name__ == "__main__":
    main()
