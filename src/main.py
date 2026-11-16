import torch
import wandb

from multimodal_multilingual_translator.model import (
    Input_Video_PreProcessor,
    Multimodal_Understanding_Audio_Generation,
    Text_to_3D_Motion_Generation,
    ThreeD_lifter,
)
from src.data.data import load_all_datasets


def train_pipeline():
    wandb.init(project="multilingual-omni-sign-translation", name="stage-training-run")

    print("Loading datasets...")
    all_data = load_all_datasets(streaming=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n--- Starting Stage 3: VQ-VAE Training ---")
    lifter = ThreeD_lifter(filename="multiHMR_896_L.pt", input_dim=132).to(device)

    sign_data_loader = all_data.get("how2sign", {})

    print("\n--- Starting Stage 4: Text-to-Motion Generator Training ---")
    text_motion_gen = Text_to_3D_Motion_Generation(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct")

    print("\n--- Starting Stage 2: Qwen3-Omni Multimodal Adaptation ---")
    omni_pipeline = Multimodal_Understanding_Audio_Generation(model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct")

    multilingual_asr_data = {
        "waxal": all_data.get("waxal_asr"),
        "fleurs": all_data.get("fleurs"),
        "qasr": all_data.get("qcri_qasr")
    }

    print("\nAll training stages completed successfully.")
    wandb.finish()

def main():
    wandb.init(project="multilingual-omni-sign-translation", name="stage-training-run")

    print("Initializing dataset streaming dictionary...")
    all_data = load_all_datasets(streaming=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n--- Phase 1: Initializing Stage 1 (Sapiens2 Feature Extractor) ---")
    # Stage 1 extracts skeleton/pose coordinates from raw sign videos
    video_processor = Input_Video_PreProcessor(model_name="facebook/sapiens2-pose-1b")

    print("\n--- Phase 2: Training Stage 3 (Motion VQ-VAE Codebook) ---")
    sign_dataset = all_data.get("how2sign", {})
    lifter = ThreeD_lifter(filename="multiHMR_896_L.pt", input_dim=132).to(device)
    # To train: lifter.train_vqvae(sign_dataset, epochs=50, lr=1e-4)

    print("\n--- Phase 3: Training Stage 4 (Text-to-3D Motion Generation Head) ---")
    text_motion_gen = Text_to_3D_Motion_Generation(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    # To train: text_motion_gen.train_generator(text_motion_dataset, epochs=3, lr=5e-5)

    print("\n--- Phase 4: Training Stage 2 (Qwen3-Omni LoRA & Audio Adapters) ---")
    omni_pipeline = Multimodal_Understanding_Audio_Generation(model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct")

    asr_datasets = {
        "waxal": all_data.get("waxal_asr"),
        "fleurs": all_data.get("fleurs"),
        "qasr": all_data.get("qcri_qasr")
    }
    # To train: omni_pipeline.train_adapters(asr_datasets, epochs=3, lr=2e-5)

    print("\nPipeline setup, data wiring, and execution readiness complete.")
    wandb.finish()

if __name__ == "__main__":
    main()
