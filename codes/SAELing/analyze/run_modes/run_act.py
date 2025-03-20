import torch
import transformers
from opensae.transformer_with_sae import TransformerWithSae, InterventionConfig
from opensae import OpenSae
from collections import Counter, defaultdict


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
    lines = test_text.strip().split("\n")
    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        print("[extract_data] No non-empty lines.")
        return []
        
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    encodings = tokenizer(
        lines,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    structured_data = model.extract_data(encodings, tokenizer)

    # ========== Part B: Test analyze_data (activation mode) ==========  
    analyze_activation = model.analyze_data(
        structured_data,
        mode="activation",  # "activation" mode
        top_k=100,
        noise_bases=set()
    )

    # Check if "all_sorted" is available in activation results
    if "all_sorted" in analyze_activation:
        # Extract all base vectors and their activation values
        all_sorted = analyze_activation["all_sorted"]

        # ========== Find positions of vectors in 'to_see' list ==========
        to_see = [167285]  # Base vector IDs to check

        vector_positions = {}
        for b_id in to_see:
            # Find position of each base vector in all_sorted
            position = next((index + 1 for index, (base_id, _) in enumerate(all_sorted) if base_id == b_id), None)

            # Record position or None if not found
            vector_positions[b_id] = position if position else None

        print("[VECTOR POSITIONS] Rank positions of to_see vectors:")
        for b_id, pos in vector_positions.items():
            if pos is None:
                print(f"Base vector {b_id} is not in sorted list.")
            else:
                print(f"Base vector {b_id} is ranked at position {pos}.")

        # ========== Part C: Test visualize ==========  
        output_html_path = "/path/to/output_visualization.html"
        model.visualize(structured_data, [base_id for base_id, _ in all_sorted], output_html=output_html_path)
        print(f"[VISUALIZE] Saved to '{output_html_path}'")
    else:
        print("[VISUALIZE] No valid activation results for visualization.")


if __name__ == "__main__":
    main()
