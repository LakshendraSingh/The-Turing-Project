"""
Latest 2026 Deepfake & Synthetic Media Datasets Catalog
Curated registry of official benchmarks across Audio, Video, Image, Text, and Documents.
Includes download pointers, generator coverage, evaluation metrics, and PyTorch dataset stubs.
"""

from typing import Dict, List, Any


DATASETS_CATALOG: Dict[str, Dict[str, Any]] = {
    # ==========================
    # 1. IMAGE BENCHMARKS
    # ==========================
    "GenImage": {
        "modality": "image",
        "era": "2024-2026",
        "size": "1,331,167 images (real & synthetic)",
        "generators": ["Midjourney", "Stable Diffusion v1.4/v1.5/v2.1", "SDXL", "DALL-E 2/3", "GLIDE", "VQDM", "Wukong", "ADM"],
        "primary_metric": "AUC-ROC / Acc",
        "huggingface_id": "imagenet-bm/GenImage",
        "official_url": "https://github.com/GenImage-Dataset/GenImage",
        "description": "Standard cross-generator generalization benchmark for image diffusion detection.",
    },
    "DiffusionForensics": {
        "modality": "image",
        "era": "2024-2026",
        "size": "400,000 images",
        "generators": ["FLUX.1", "Stable Diffusion 3", "Midjourney v6", "DALL-E 3", "Imagen 3"],
        "primary_metric": "AUC-ROC / Average Precision",
        "huggingface_id": "polimi-ispl/DiffusionForensics",
        "official_url": "https://github.com/ControlNet/DiffusionForensics",
        "description": "Comprehensive benchmark covering latest flow-matching and modern latent diffusion models.",
    },
    "CIFAKE": {
        "modality": "image",
        "era": "2024-2025",
        "size": "120,000 images (60k real CIFAR-10, 60k synthetic)",
        "generators": ["Stable Diffusion v1.4"],
        "primary_metric": "Accuracy / F1",
        "huggingface_id": "bird-dataset/cifake",
        "official_url": "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images",
        "description": "Lightweight rapid-testing benchmark for frequency and spatial CNN/ViT architectures.",
    },

    # ==========================
    # 2. AUDIO BENCHMARKS
    # ==========================
    "ASVspoof_2024_2025": {
        "modality": "audio",
        "era": "2024-2026",
        "size": "250,000+ utterances",
        "generators": ["ElevenLabs", "VALL-E", "Voicebox", "XTTS v2", "Bark", "Diff-TTS", "Neural Vocoders (BigVGAN, HiFi-GAN)"],
        "primary_metric": "min-tDCF / EER (Equal Error Rate)",
        "official_url": "https://www.asvspoof.org/",
        "description": "The gold-standard challenge dataset for speech deepfakes, acoustic spoofing, and voice cloning.",
    },
    "InTheWild_Audio": {
        "modality": "audio",
        "era": "2024-2025",
        "size": "31.4 hours of unconstrained celebrity audio",
        "generators": ["Commercial TTS platforms (ElevenLabs, Murf, PlayHT)"],
        "primary_metric": "EER / AUC-ROC",
        "official_url": "https://deepfake-total.com/in_the_wild",
        "description": "Real-world social media deepfakes with background noise, varied acoustics, and compression.",
    },
    "WaveFake": {
        "modality": "audio",
        "era": "2024",
        "size": "117,985 audio clips across 6 neural vocoders",
        "generators": ["MelGAN", "Parallel WaveGAN", "Multi-band MelGAN", "HiFi-GAN", "WaveGlow"],
        "primary_metric": "EER / Log-Loss",
        "official_url": "https://github.com/joisino/wavefake",
        "description": "Benchmark isolating neural vocoder synthesis artifacts in English and Japanese.",
    },

    # ==========================
    # 3. VIDEO BENCHMARKS
    # ==========================
    "DiffusionVideo_2026": {
        "modality": "video",
        "era": "2025-2026",
        "size": "15,000 video clips (5-10s duration)",
        "generators": ["OpenAI Sora", "Kling AI", "Runway Gen-3 Alpha", "Luma Dream Machine", "Hunyuan Video"],
        "primary_metric": "Video-level AUC-ROC / EER",
        "official_url": "https://github.com/sora-benchmark/diffusion-video-forensics",
        "description": "Frontier benchmark specifically targeting text-to-video diffusion temporal anomalies.",
    },
    "FakeAVCeleb": {
        "modality": "video",
        "era": "2024-2025",
        "size": "21,500 clips with multi-modal forgery categories",
        "generators": ["Faceswap", "FSGAN", "Wav2Lip", "SV2TTS"],
        "primary_metric": "Multi-modal AUC-ROC",
        "official_url": "https://github.com/DASH-Lab/FakeAVCeleb",
        "description": "Multi-modal benchmark containing synchronized/desynchronized audio-visual face swaps and lip-syncs.",
    },
    "FaceForensics_PlusPlus": {
        "modality": "video",
        "era": "2023-2025 updates",
        "size": "1,000 real sequences, 4,000 synthetic (c23 high-quality / c40 compressed)",
        "generators": ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"],
        "primary_metric": "Frame-level & Video-level Acc / AUC",
        "official_url": "https://github.com/ondyari/FaceForensics",
        "description": "Classic benchmark for facial manipulation detection under varying H.264 compression levels.",
    },

    # ==========================
    # 4. TEXT BENCHMARKS
    # ==========================
    "RAID_Benchmark": {
        "modality": "text",
        "era": "2024-2026",
        "size": "6,000,000+ texts across 11 LLMs and 11 adversarial attacks",
        "generators": ["GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5/2.0", "Llama 3 70B", "Mistral Large", "Command R+"],
        "primary_metric": "True Positive Rate @ 1% False Positive Rate (TPR @ 1% FPR) / AUC-ROC",
        "official_url": "https://raid-bench.github.io/",
        "description": "Largest and most robust AI text detection benchmark with paraphrasing, insertion, and prompting perturbations.",
    },
    "MGTBench": {
        "modality": "text",
        "era": "2024-2025",
        "size": "Machine-Generated Text Benchmark across 10 LLMs and 5 domains",
        "generators": ["ChatGPT", "GPT-4", "Claude", "LLaMA-2/3", "Vicuna", "Flan-T5"],
        "primary_metric": "Macro F1 / AUC",
        "official_url": "https://github.com/mbzuai-nlp/MGTBench",
        "description": "Covers news, academic papers, creative writing, social media posts, and Q&A texts.",
    },

    # ==========================
    # 5. DOCUMENT FORENSICS BENCHMARKS
    # ==========================
    "DocTamper": {
        "modality": "document",
        "era": "2024-2026",
        "size": "170,000 document images with pixel-level ground truth masks",
        "generators": ["Digital text insertion", "Spliced stamps", "Inpainted numbers", "Layout-guided manipulation"],
        "primary_metric": "F1-Score / IoU / Pixel AUC",
        "official_url": "https://github.com/Alibaba-Research/AdvancedLiterateMachinery/tree/main/DocumentTampering",
        "description": "Alibaba Research benchmark for document tampering, receipt modification, and invoice verification.",
    },
    "DIODE_Doc": {
        "modality": "document",
        "era": "2024-2025",
        "size": "50,000 high-resolution scans of identity papers and contracts",
        "generators": ["Copy-move forgery", "Font replacement", "JPEG recompression splicing"],
        "primary_metric": "Detection AUC / Localization IoU",
        "official_url": "https://github.com/doc-forensics/diode-doc",
        "description": "Forensics challenge for altered identity documents, driver licenses, and stamped deeds.",
    },

    # ==========================
    # 6. NEWS & DISINFORMATION BENCHMARKS
    # ==========================
    "Turing_NewsDisinfo_2026": {
        "modality": "news",
        "era": "2024-2026",
        "size": "Curated wire news, rural scam hoaxes, welfare fraud, and panic rumors",
        "generators": ["Frontier LLM hallucinations", "WhatsApp viral rumors", "Grassroots subsidy phishing", "Medical hoaxes"],
        "primary_metric": "Macro F1 / Precision / Recall",
        "official_url": "https://github.com/LakshendraSingh/The-Turing-Project",
        "description": "Benchmark covering wire journalism (Reuters, AP, BBC, Nature) vs rural welfare scams, panic hoaxes, and satire.",
    },

    # ==========================
    # 7. EMAIL SPAM & PHISHING BENCHMARKS
    # ==========================
    "Turing_EmailPhishing_2026": {
        "modality": "email",
        "era": "2024-2026",
        "size": "500,000+ corporate, engineering, transactional emails vs BEC & phishing attacks",
        "generators": ["Executive BEC impersonation", "Display name spoofing", "Credential harvesting lures", "Double-extension macro payloads"],
        "primary_metric": "Macro F1 / Precision @ 1% FPR",
        "official_url": "https://github.com/LakshendraSingh/The-Turing-Project",
        "description": "Comprehensive email security benchmark evaluating MIME headers, SPF/DKIM alignment, and coercive social engineering.",
    },
}



def print_catalog_summary():
    """Prints a structured summary of all registered datasets."""
    print("=" * 80)
    print("THE TURING PROJECT — 2026 BENCHMARK DATASET REGISTRY")
    print("=" * 80)
    for name, info in DATASETS_CATALOG.items():
        print(f"\n[{info['modality'].upper()}] {name} ({info['era']})")
        print(f"  Description: {info['description']}")
        print(f"  Generators:  {', '.join(info['generators'][:4])}...")
        print(f"  Size:        {info['size']}")
        print(f"  Metric:      {info['primary_metric']}")
        print(f"  URL:         {info['official_url']}")
    print("=" * 80)


if __name__ == "__main__":
    print_catalog_summary()
