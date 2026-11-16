import torch
import wandb

from src.data.data import load_all_datasets
from src.multimodal_multilingual_translator.model import (
    Input_Video_PreProcessor,
    Multimodal_Understanding_Audio_Generation,
    Text_to_3D_Motion_Generation,
    ThreeD_lifter,
)


def main(train:bool=False):
    if train:
        wandb.init(project="multilingual-omni-sign-translation", name="stage-training-run")

        print("Initializing dataset streaming dictionary...")
        all_data = load_all_datasets(streaming=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("\n--- Phase 1: Initializing Stage 1 (Sapiens2 Feature Extractor) ---")
        video_processor = Input_Video_PreProcessor(model_name="facebook/sapiens2-pose-1b")

        print("\n--- Phase 2: Training Stage 3 (Motion VQ-VAE Codebook) ---")
        sign_dataset = all_data.get("how2sign", {})
        lifter = ThreeD_lifter(filename="multiHMR_896_L.pt", input_dim=132).to(device)
        lifter.train_vqvae(sign_dataset, epochs=50, lr=1e-4)

        print("\n--- Phase 3: Training Stage 4 (Text-to-3D Motion Generation Head) ---")
        text_motion_dataset = {
            "how2sign": all_data.get("how2sign", {}),
            "signverse": all_data.get("signverse", {})
        }

        text_motion_gen = Text_to_3D_Motion_Generation(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct")
        text_motion_gen.train_generator(text_motion_dataset, epochs=3, lr=5e-5)

        print("\n--- Phase 4: Training Stage 2 (Qwen3-Omni LoRA & Audio Adapters) ---")
        omni_pipeline = Multimodal_Understanding_Audio_Generation(model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct")

        asr_datasets = {
            "waxal": all_data.get("waxal_asr"),
            "fleurs": all_data.get("fleurs"),
            "qasr": all_data.get("qcri_qasr")
        }
        omni_pipeline.train_adapters(asr_datasets, epochs=3, lr=2e-5)

        print("\nPipeline setup, data wiring, and execution readiness complete.")
        wandb.finish()

    print("ran successfully")
