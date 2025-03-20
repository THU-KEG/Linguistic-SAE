import os
import torch
import transformers
from opensae.transformer_with_sae import TransformerWithSae
from opensae import OpenSae


def process_experiment(input_text_path, output_html_path, model, tokenizer):
    # Read input text
    with open(input_text_path, "r", encoding="utf-8") as file:
        test_text = file.read()
    print(f"[INPUT TEXT] {input_text_path}")

    # Preprocess and tokenize text
    lines = [ln.strip() for ln in test_text.strip().split("\n") if ln.strip()]
    if not lines:
        print("[extract_data] No non-empty lines.")
        return

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    encodings = tokenizer(lines, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Extract features
    structured_data = model.extract_data(encodings, tokenizer)

    # Analyze data (Top-k activation vectors)
    analyze_freq = model.analyze_data(
        structured_data,
        mode="frequency",
        top_k=40,
        token_index=0,
        noise_bases=set()
    )
    if "top_k_results" in analyze_freq:
        base_vector_indices = [result[0] for result in analyze_freq["top_k_results"]]

        # Visualize results
        model.visualize(structured_data, base_vector_indices, output_html=output_html_path)
        print(f"[VISUALIZE] Saved to '{output_html_path}'")
    else:
        print("[VISUALIZE] No valid top_k_results.")


def main():
    device = "cuda:7"  # Set device

    # Input and output directories (masked)
    input_dir = "/path/to/data"
    output_dir = "/path/to/output"

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-trained SAE model
    sae = OpenSae.from_pretrained("/path/to/sae_checkpoint")

    # Wrap LLAMA model with TransformerWithSae
    model = TransformerWithSae(
        "/path/to/llama_model",
        sae,
        device
    )

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("/path/to/llama_model")

    # Process all .txt files
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".txt"):
            # Build file paths
            experiment_name = os.path.splitext(file_name)[0]
            input_text_path = os.path.join(input_dir, file_name)
            output_html_path = os.path.join(output_dir, f"{experiment_name}.html")

            # Process current experiment
            process_experiment(input_text_path, output_html_path, model, tokenizer)


if __name__ == "__main__":
    main()
