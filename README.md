# multimodal_multilingual_translator
## Description
This is a multilingual translator that includes an ASR model to translate spoken text to any number of languages and a visual model to include multilingual signed languages.

Detailed Model Breakdown


1. Input Video Pre-Processor

• Model: MediaPipe Holistic (Google) or facebook/sapiens-pose-1b.
• Role: Runs on your incoming sign language video to draw clean hand, finger, and facial keypoints over the frames before feeding them into Qwen.

2. Multimodal Understanding & Audio Generation (Core Engine)

• Model: Qwen/Qwen3-Omni-30B-A3B-Instruct (or Qwen/Qwen2.5-Omni-7B for ~16 GB VRAM).
• Role: Takes the landmark-enhanced video and spoken audio simultaneously, then streams out the translated multilingual text and spoken audio waveform. Requires a LoRA adapter fine-tuned on SignerX/SignVerse-2M for accurate sign comprehension.

3. Offline 3D Data Lifter (Training Prep Only)

• Model: caizhongang/SMPLer-X (smpler_x_h32 checkpoint) + the official SMPL-X parametric body model files.
• Role: You run this once offline on the SignerX/SignVerse-2M videos to extract 3D body poses, 30-joint finger rotations, and facial expression blendshapes (betas, body_pose, left_hand_pose, right_hand_pose, expression).

4. Text-to-3D Motion Generator (Trained Model)

• Model Architecture: Adapt MDM (Motion Diffusion Model), T2M-GPT, or a Transformer Sequence-to-Sequence network using a frozen multilingual text encoder (like Qwen/Qwen3-Embedding-0.6B or google/mt5-base).
• Role: Takes Qwen's output text and predicts a continuous sequence of 3D SMPL-X joint rotations and facial blendshapes at 24–30 FPS.

5. Final 3D Video Renderer (Choose One)

• Option A — Headless Blender + smplx Add-on (Fast & Crisp):
	• Models Needed: None (uses standard 3D mesh rendering via Python bpy).
	• Role: Applies the predicted SMPL-X rotations to a rigged 3D character and renders an .mp4 video with zero finger distortion and minimal GPU overhead.
• Option B — fudan-generative-ai/champ (Photorealistic Human):
	• Models Needed: fudan-generative-ai/champ + runwayml/stable-diffusion-v1-5 + stabilityai/sd-vae-ft-mse.
	• Role: Projects the 3D SMPL-X mesh into depth, normal, and semantic maps, then uses diffusion to render a photorealistic human signer matching a reference photo.
Would you like the training script structure for Stage 4 (Text-to-SMPL-X) or the Python setup for rendering SMPL-X animations in Blender vs. Champ?
