import torch
import transformers
from opensae.transformer_with_sae import TransformerWithSae, InterventionConfig
from opensae import OpenSae


def main():
    device = "cuda:0"

    # ===== 1. Load SAE =====
    sae = OpenSae.from_pretrained("/path/to/sae_model_checkpoint")

    # ===== 2. Wrap LLAMA with TransformerWithSae =====
    model = TransformerWithSae(
        "/path/to/llama_model",
        sae,
        device
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained("/path/to/llama_model")

    # -----------------------------------------------------------------------------    
    # Read test input
    with open("/path/to/test_input.txt", "r", encoding="utf-8") as file:
        test_text = file.read()

    print("[INPUT TEXT]\n", test_text)

    # ========== Part A: Test extract_data ========== 
    # Extract base vector activations from text and create structured data
    lines = test_text.strip().split("\n")
    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        print("[extract_data] No non-empty lines.")
        return []

    # Ensure tokenizer pad_token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Tokenize all lines in batch mode
    encodings = tokenizer(
        lines,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    structured_data = model.extract_data(encodings, tokenizer)

    # ========== Part B: Test analyze_data ========== 
    # Frequency mode
    analyze_freq = model.analyze_data(
        structured_data,
        mode="frequency",
        top_k=60,
        token_index=0,
        noise_bases=set()
    )

    # Check if top_k_results is available
    if "top_k_results" in analyze_freq:
        # Extract base vector IDs from top_k results
        base_vector_indices = [result[0] for result in analyze_freq["top_k_results"]]
        to_see = [75327]

    # ========== Part C: Test visualize ========== 
    # Visualize structured data and top_k base vector IDs
    output_html_path = "/path/to/output_visualization.html"
    to_see_html_path = "/path/to/output_to_see.html"
    model.visualize(structured_data, base_vector_indices, output_html=output_html_path)
    model.visualize(structured_data, to_see, output_html=to_see_html_path)
    print(f"[VISUALIZE] Visualization saved to '{output_html_path}'")
    else:
        print("[VISUALIZE] No valid top_k_results.")

if __name__ == "__main__":
    main()
