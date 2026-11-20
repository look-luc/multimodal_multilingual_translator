import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
    Sapiens2ForPoseEstimation,
)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class Input_Video_PreProcessor:
    def __init__(self, model_name: str = "facebook/sapiens2-pose-1b") -> None:
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Sapiens2ForPoseEstimation.from_pretrained(model_name, device_map=DEVICE)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        flat_input = inputs.view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(inputs.shape)

        vq_loss = F.mse_loss(quantized.detach(), inputs)
        commitment_loss = F.mse_loss(quantized, inputs.detach())
        loss = vq_loss + self.commitment_cost * commitment_loss

        quantized = inputs + (quantized - inputs).detach()

        return quantized.permute(0, 2, 1).contiguous(), loss, encoding_indices

class ThreeD_lifter(nn.Module):
    def __init__(self, filename: str, repo_id: str = "naver/multi-hmr", input_dim: int = 132) -> None:
        super().__init__()
        try:
            self.ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="./pretrained_models")
        except Exception:
            print("Pretrained checkpoint path optional or missing; initializing fresh VQ-VAE.")

        self.vq_codebook_size = 512
        self.latent_dim = 512

        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, self.latent_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(self.latent_dim),
        )

        self.vq_vae = VectorQuantizer(self.vq_codebook_size, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.latent_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, input_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        enc_out = self.encoder(x)
        quantized, vq_loss, indices = self.vq_vae(enc_out)
        recon = self.decoder(quantized)
        return recon, vq_loss, indices

    def train_vqvae(self, motion_dataset, epochs: int = 50, lr: float = 1e-4):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.to(DEVICE)
        self.train()

        for epoch in range(epochs):
            total_recon_loss = 0.0
            total_vq_loss = 0.0
            for batch in motion_dataset:
                poses = batch["skeleton_poses"].to(DEVICE)

                recon, vq_loss, _ = self.forward(poses)
                recon_loss = F.l1_loss(recon, poses)

                loss = recon_loss + vq_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_recon_loss += recon_loss.item()
                total_vq_loss += vq_loss.item()

            print(f"Epoch [{epoch+1}/{epochs}] | Recon Loss: {total_recon_loss:.4f} | VQ Loss: {total_vq_loss:.4f}")

class Multimodal_Understanding_Audio_Generation:
    def __init__(self, model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct") -> None:
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_path, device_map=DEVICE)
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        self.audio_adapter = torch.nn.Linear(768, self.model.config.hidden_size).to(DEVICE)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.model = get_peft_model(self.model, peft_config)

    def train_adapters(self, datasets_dict, epochs: int = 3, lr: float = 2e-5):
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.audio_adapter.parameters()),
            lr=lr
        )
        self.model.train()

        for epoch in range(epochs):
            print(f"Starting Epoch {epoch+1}/{epochs} for Qwen3-Omni Adapters...")
            for dataset_name, lang_dict in datasets_dict.items():
                if not lang_dict:
                    continue
                for lang_name, dataset in lang_dict.items():
                    for batch in dataset:
                        audio_arrays = batch.get("audio", {}).get("array")
                        transcripts = batch.get("text") or batch.get("transcription")

                        if audio_arrays is None or transcripts is None:
                            continue

                        inputs = self.processor(
                            text=transcripts,
                            audios=audio_arrays,
                            return_tensors="pt",
                            padding=True
                        ).to(DEVICE)

                        outputs = self.model(**inputs, labels=inputs["input_ids"])
                        loss = outputs.loss

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

class Text_to_3D_Motion_Generation:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE)
        self.motion_head = torch.nn.Linear(self.model.config.hidden_size, 512).to(DEVICE)

    def train_generator(self, text_motion_dataset, epochs: int = 3, lr: float = 5.0e-5):
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.motion_head.parameters()),
            lr=lr
        )
        self.model.train()
        self.motion_head.train()

        for epoch in range(epochs):
            for batch in text_motion_dataset:
                input_ids = batch["input_ids"].to(DEVICE)
                target_motion_tokens = batch["motion_tokens"].to(DEVICE)

                outputs = self.model(input_ids=input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]

                logits = self.motion_head(hidden_states)

                loss = F.cross_entropy(
                    logits.view(-1, 512),
                    target_motion_tokens.view(-1)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
