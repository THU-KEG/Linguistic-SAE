import torch
import transformers
from opensae.transformer_with_sae import TransformerWithSae, InterventionConfig
from opensae import OpenSae
from collections import defaultdict


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
    # Extract base vector activations and create structured data
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

    # ========== Part B: Test analyze_data (token_frequency mode) ==========  
    analyze_token_freq = model.analyze_data(
        structured_data,
        mode="token_frequency",  # Use "token_frequency" mode
        top_k=100,
        noise_bases=set()
    )

    # Check if "all_sorted" is available
    if "all_sorted" in analyze_token_freq:
        # Extract base vectors and their activation frequency
        all_sorted = analyze_token_freq["all_sorted"]

        # ========== Find positions of vectors in 'to_see' list ==========
        to_see = [167285]  # List of base vector IDs to check

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

    else:
        print("[VISUALIZE] No valid token_frequency results found.")

if __name__ == "__main__":
    main()
