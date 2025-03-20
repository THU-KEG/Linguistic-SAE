import torch
import transformers
from opensae.transformer_with_sae import TransformerWithSae, InterventionConfig
from opensae import OpenSae

def process_experiment(input_text_path, output_text_path, model, tokenizer, experiment_name, intervention_indices, num_generations=10):
    # Read input prompt
    with open(input_text_path, "r", encoding="utf-8") as file:
        x = file.read()

    print(f"[INPUT TEXT] {input_text_path}")

    # Open output file for writing
    with open(output_text_path, "w", encoding="utf-8") as out_file:
        out_file.write(f"Experiment: {experiment_name}\n\n")
        out_file.write(f"[INPUT PROMPT]: {x}\n\n")

    for generation in range(1, num_generations + 1):
        print(f"Starting generation {generation}...")

        inputs = tokenizer(x, return_tensors="pt").to(model.device)

        # Get features
        features, _ = model(return_features=True, **inputs)

        # Append results to output file
        with open(output_text_path, "a", encoding="utf-8") as out_file:
            out_file.write(f"--- Generation {generation} ---\n")

            # Ablation: set mode, value = 0
            intervention_config = InterventionConfig(
                intervention=True,
                intervention_mode="set",
                intervention_indices=intervention_indices,
                intervention_value=0.0,
                prompt_only=False,
            )
            model.update_intervention_config(intervention_config)
            y_abl = model.generate(**inputs, max_new_tokens=500, temperature=1.0)
            output_abl = tokenizer.decode(y_abl[0], skip_special_tokens=True)
            out_file.write("[Ablation (set value=0)]:\n")
            out_file.write(output_abl + "\n\n")

            # Enhancement: set mode, value = 10
            intervention_config.intervention_value = 10.0
            model.update_intervention_config(intervention_config)
            y_enh = model.generate(**inputs, max_new_tokens=100, temperature=1.0)
            output_enh = tokenizer.decode(y_enh[0], skip_special_tokens=True)
            out_file.write("[Enhancement (set value=10)]:\n")
            out_file.write(output_enh + "\n\n")

            # Invariant: multiply mode, value = 1 (no change)
            intervention_config = InterventionConfig(
                intervention=True,
                intervention_mode="multiply",
                intervention_indices=intervention_indices,
                intervention_value=1.0,
                prompt_only=False,
            )
            model.update_intervention_config(intervention_config)
            y_inv = model.generate(**inputs, max_new_tokens=100, temperature=1.0)
            output_inv = tokenizer.decode(y_inv[0], skip_special_tokens=True)
            out_file.write("[Invariant (multiply value=1)]:\n")
            out_file.write(output_inv + "\n\n")

        print(f"Generation {generation} completed and written to {output_text_path}.")

def main():
    device = "cuda:7"

    # Paths and experiment settings
    input_text_path = "/path/to/input/causalty.txt"  # Masked path
    output_text_path = "/path/to/output/causalty.txt"  # Masked path
    experiment_name = "causalty"  # Experiment name
    intervention_indices = [223621]  # Intervention indices

    # Load pre-trained SAE model
    sae = OpenSae.from_pretrained("/path/to/sae_model_checkpoint")  # Masked path

    # Wrap LLAMA model with SAE
    model = TransformerWithSae(
        "/path/to/llama_model",  # Masked path
        sae,
        device
    )

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("/path/to/llama_model")  # Masked path

    # Run experiment with 10 generations
    process_experiment(
        input_text_path,
        output_text_path,
        model,
        tokenizer,
        experiment_name,
        intervention_indices,
        num_generations=10  # Number of generations
    )

if __name__ == "__main__":
    main()
