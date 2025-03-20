import torch
import transformers
from opensae.transformer_with_sae import TransformerWithSae
from opensae import OpenSae

def main():
    device = "cuda:0"  # Set device
    experiment_name = "ana_politeness"  # Experiment name

    # Load pre-trained SAE model
    sae = OpenSae.from_pretrained("/path/to/sae_checkpoint")

    # Wrap LLAMA model with TransformerWithSae
    model = TransformerWithSae("/path/to/llama_model", sae, device)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("/path/to/llama_model")

    # Generate input and output paths
    input_text_path = f"/path/to/data/{experiment_name}.txt"
    output_html_path = f"/path/to/output/{experiment_name}.html"

    # Read input text
    with open(input_text_path, "r", encoding="utf-8") as file:
        test_text = file.read()
    print("[INPUT TEXT]\n", test_text)

    # Process and tokenize text
    lines = [ln.strip() for ln in test_text.strip().split("\n") if ln.strip()]
    if not lines:
        print("[extract_data] No non-empty lines found.")
        return []

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    encodings = tokenizer(lines, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Extract features
    structured_data = model.extract_data(encodings, tokenizer)

    # Analyze data in frequency and activation modes
    analyze_freq = model.analyze_data(
        structured_data,
        mode="frequency",
        top_k=200,
        token_index=0,
        noise_bases=set()  # No noise filtering, manual filtering later
    )

    analyze_act = model.analyze_data(
        structured_data,
        mode="activation",
        top_k=1000,
        token_index=0,
        noise_bases=set()
    )

    # Define noise bases
    noise_bases_list = [58881, 231964, 248350, 103461, 80939, 44079, 135220, 183356, 162881, 129610, 26190, 152664, 255073, 126566, 192102, 151146, 83565, 23674, 189058, 238724, 139396, 59527, 2195, 190103, 112287, 157343, 10918, 46249, 29881, 126147, 53443, 111310, 50907, 11997, 13538, 167141, 77038, 127732, 192259, 245508, 13079, 181528]
    noise_bases = set(noise_bases_list)

    # Define target vector
    target_vector = 149519

    def compute_ranks(analyze_result, mode_name):
        """
        Compute rank of target vector in analyze_result top_k_results:
        - rank_original: position of target_vector in original list (1-indexed)
        - rank_no_noise: position of target_vector after removing noise bases (1-indexed)
        """
        if "top_k_results" not in analyze_result:
            print(f"[{mode_name}] No valid top_k_results.")
            return None, None

        # Get base vector indices
        base_vector_indices = [result[0] for result in analyze_result["top_k_results"]]

        if target_vector in base_vector_indices:
            rank_original = base_vector_indices.index(target_vector) + 1
        else:
            rank_original = None

        # Remove noise bases
        base_vector_indices_no_noise = [vec for vec in base_vector_indices if vec not in noise_bases]
        if target_vector in base_vector_indices_no_noise:
            rank_no_noise = base_vector_indices_no_noise.index(target_vector) + 1
        else:
            rank_no_noise = None

        return rank_original, rank_no_noise

    # Calculate ranks in both frequency and activation modes
    freq_rank_original, freq_rank_no_noise = compute_ranks(analyze_freq, "frequency")
    act_rank_original, act_rank_no_noise = compute_ranks(analyze_act, "activation")

    # Output results
    print("Target vector 75327 ranks in frequency mode:")
    print("Original rank:", freq_rank_original)
    print("Rank without noise:", freq_rank_no_noise)

    print("\nTarget vector 75327 ranks in activation mode:")
    print("Original rank:", act_rank_original)
    print("Rank without noise:", act_rank_no_noise)

if __name__ == "__main__":
    main()
