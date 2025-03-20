import torch
import transformers
from opensae.transformer_with_sae import TransformerWithSae
from opensae import OpenSae
import gc

def main():
    device = "cuda:0"  # Set device
    experiment_name = "time"  # Experiment name

    # Define layers to analyze
    layers = ["27", "28", "29", "30", "31"]

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("/path/to/llama_model")

    # Generate input text path
    input_text_path = f"/path/to/data/{experiment_name}.txt"

    # Read input text
    try:
        with open(input_text_path, "r", encoding="utf-8") as file:
            test_text = file.read()
        print("[INPUT TEXT]\n", test_text)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {input_text_path}")
        return
    except Exception as e:
        print(f"[ERROR] Error reading file: {e}")
        return

    # Preprocess and tokenize
    lines = [ln.strip() for ln in test_text.strip().split("\n") if ln.strip()]
    if not lines:
        print("[extract_data] No non-empty lines found.")
        return

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    try:
        encodings = tokenizer(lines, return_tensors="pt", padding=True, truncation=True).to(device)
    except Exception as e:
        print(f"[ERROR] Tokenization error: {e}")
        return

    for layer in layers:
        print(f"\nProcessing layer {layer}...")

        # Load pre-trained SAE model for the layer
        sae_path = f"/path/to/sae_checkpoint/_{layer}"
        try:
            sae = OpenSae.from_pretrained(sae_path)
            print(f"[LOAD SAE] Loaded SAE model: {sae_path}")
        except Exception as e:
            print(f"[ERROR] Unable to load SAE model: {sae_path}. Error: {e}")
            continue

        # Wrap LLAMA model with TransformerWithSae
        try:
            model = TransformerWithSae("/path/to/llama_model", sae, device)
            print(f"[INIT MODEL] Initialized TransformerWithSae model.")
        except Exception as e:
            print(f"[ERROR] Unable to initialize TransformerWithSae. Error: {e}")
            continue

        # Extract features
        try:
            structured_data = model.extract_data(encodings, tokenizer)
            print(f"[EXTRACT DATA] Extracted data for layer {layer}.")
        except Exception as e:
            print(f"[ERROR] Data extraction failed. Error: {e}")
            continue

        # Analyze data (Top-k activation vectors)
        try:
            analyze_freq = model.analyze_data(
                structured_data,
                mode="frequency",
                top_k=80,
                token_index=0,
                noise_bases=set()
            )
            print(f"[ANALYZE DATA] Data analysis completed for layer {layer}.")
        except Exception as e:
            print(f"[ERROR] Data analysis failed. Error: {e}")
            continue

        if "top_k_results" in analyze_freq:
            base_vector_indices = [result[0] for result in analyze_freq["top_k_results"]]
            print(f"[TOP K] Extracted {len(base_vector_indices)} Top-k activation vectors.")
        else:
            print(f"[VISUALIZE] No valid top_k_results for layer {layer}.")
            # Clean up and move to next layer
            del model, sae, analyze_freq, structured_data
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # Visualize the results
        output_html_path = f"/path/to/output/{experiment_name}_{layer}.html"
        try:
            model.visualize(structured_data, base_vector_indices, output_html=output_html_path)
            print(f"[VISUALIZE] Saved visualization for layer {layer} to '{output_html_path}'")
        except Exception as e:
            print(f"[ERROR] Visualization failed. Error: {e}")
        finally:
            # Clean up resources
            del model, sae, analyze_freq, structured_data, base_vector_indices
            torch.cuda.empty_cache()
            gc.collect()
            print(f"[COMPLETE] Layer {layer} processing complete.")

    print("\nAnalysis and visualization for all layers completed.")

if __name__ == "__main__":
    main()
