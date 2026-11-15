import torch
from huggingface_hub import hf_hub_download
from qwen_omni_utils import process_mm_info
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
    Sapiens2ForPoseEstimation,
)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class Input_Video_PreProcessor: #stage 1
    def __init__(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained("facebook/sapiens2-pose-1b")
        self.model = Sapiens2ForPoseEstimation.from_pretrained(
            "facebook/sapiens2-pose-1b", device_map=DEVICE
        )

class Multimodal_Understanding_Audio_Generation: #stage 2
    def __init__(self) -> None:
        self.model_path = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype="auto",
            device_map=DEVICE,
            attn_implementation="flash_attention_2",
        )

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)

class ThreeD_lifter: #stage 3
    def __init__(self) -> None:
        self.ckpt_path = hf_hub_download(
            repo_id="naver/multi-hmr",
            filename="multiHMR_896_L.pt",
            local_dir="./pretrained_models",
            local_dir_use_symlinks=False
        )

class Text_to_3D_Motion_Generattion: #stage 4
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Embedding-0.6B", device_map=DEVICE)
