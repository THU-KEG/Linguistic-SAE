import torch
import transformers
from opensae.transformer_with_sae import TransformerWithSae
from opensae import OpenSae


def main():
    device = "cuda:0"  # Set device
    experiment_name = "placemini2"  # Experiment name

    # Load pre-trained SAE model
    sae = OpenSae.from_pretrained("/path/to/sae_model_checkpoint")

    # Wrap LLAMA model with SAE
    model = TransformerWithSae(
        "/path/to/llama_model",
        sae,
        device
    )

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("/path/to/llama_model")

    # Paths for input and output
    input_text_path = f"/path/to/data/{experiment_name}.txt"
    output_html_path = f"/path/to/output/{experiment_name}.html"

    # Read input text
    with open(input_text_path, "r", encoding="utf-8") as file:
        test_text = file.read()
    print("[INPUT TEXT]\n", test_text)

    # Preprocess text and tokenize
    lines = [ln.strip() for ln in test_text.strip().split("\n") if ln.strip()]
    if not lines:
        print("[extract_data] No non-empty lines.")
        return []

    # Prepare tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    encodings = tokenizer(lines, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Extract features
    structured_data = model.extract_data(encodings, tokenizer)

    # Analyze data
    analyze_freq = model.analyze_data(
        structured_data,
        mode="frequency",
        top_k=0,
        token_index=0,
        noise_bases=set()
    )

    if "top_k_results" in analyze_freq:
        # Get base vector indices
        base_vector_indices = [result[0] for result in analyze_freq["top_k_results"]]

        # Add manual indices
        manual_additions = [229581]
        base_vector_indices.extend(manual_additions)

        # Remove duplicates
        base_vector_indices = list(set(base_vector_indices))

        # Visualize results
        model.visualize(structured_data, base_vector_indices, output_html=output_html_path)
        print(f"[VISUALIZE] Saved to '{output_html_path}'")
    else:
        print("[VISUALIZE] No valid results.")


if __name__ == "__main__":
    main()
