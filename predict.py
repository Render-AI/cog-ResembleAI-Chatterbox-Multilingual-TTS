"""
Chatterbox Multilingual TTS - Replicate Cog Implementation

High-quality multilingual text-to-speech synthesis supporting 23 languages.
Generates natural-sounding speech with optional reference audio for voice cloning.

Model: ResembleAI Chatterbox Multilingual
Languages: Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek,
          Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish,
          Portuguese, Russian, Spanish, Swedish, Swahili, Turkish

Author: ResembleAI
License: MIT
"""

import gc
import os
import subprocess
import tempfile
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from scipy.io import wavfile
from cog import BasePredictor, Input, Path as CogPath

# Configure model cache directory for all ML frameworks
MODEL_CACHE = "model_cache"
BASE_URL = "https://weights.replicate.delivery/default/ResembleAI-Chatterbox-Multilingual-TTS/model_cache/"

# Set environment variables for model caching - needs to happen early
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

# Suppress specific warnings that don't affect functionality
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message=".*torchvision.datapoints.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES


def download_weights(url: str, dest: str) -> None:
    """Download model weights using pget with proper error handling."""
    start = time.time()
    print(f"[!] Initiating download from URL: {url}")
    print(f"[~] Destination path: {dest}")
    
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
        print(f"[+] Download completed in: {time.time() - start:.2f} seconds")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to download weights from {url}. "
            f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )


def set_seed(seed: int, device: str) -> None:
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class Predictor(BasePredictor):
    """Chatterbox Multilingual TTS Predictor for Replicate"""

    def setup(self) -> None:
        """Load the multilingual TTS model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing Chatterbox Multilingual TTS on {self.device}")
        
        # Create model cache directory if it doesn't exist
        os.makedirs(MODEL_CACHE, exist_ok=True)

        # Model files to download
        model_files = [
            ".cache.tar",
            ".gitattributes", 
            "README.md",
            "conds.pt",
            "mtl_tokenizer.json",
            "s3gen.pt",
            "t3_alllang.safetensors",
            "ve.pt",
            "version.txt",
            "version_diffusers_cache.txt",
        ]

        # Download weights using pget if they don't exist
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            
            # Skip if file exists (for tar files, check the extracted directory)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)
        
        # Optimize PyTorch settings for better performance
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        
        # Load the TTS model
        self.model = ChatterboxMultilingualTTS.from_pretrained(self.device)
        print(f"✅ Model loaded successfully")
        print(f"📋 Supporting {len(SUPPORTED_LANGUAGES)} languages")

    def predict(
        self,
        text: str = Input(
            description="Text to synthesize into speech (maximum 300 characters)"
        ),
        language: str = Input(
            description="Language for synthesis",
            choices=list(SUPPORTED_LANGUAGES.keys()),
            default="en"
        ),
        reference_audio: Optional[CogPath] = Input(
            description="Reference audio file for voice cloning (optional). If not provided, uses default voice for the selected language.",
            default=None
        ),
        exaggeration: float = Input(
            description="Controls speech expressiveness (0.25-2.0, neutral=0.5, extreme values may be unstable)",
            ge=0.25,
            le=2.0,
            default=0.5
        ),
        temperature: float = Input(
            description="Controls randomness in generation (0.05-5.0, higher=more varied)",
            ge=0.05,
            le=5.0,
            default=0.8
        ),
        cfg_weight: float = Input(
            description="CFG/Pace weight controlling generation guidance (0.2-1.0). Use 0.5 for balanced results, 0 for language transfer",
            ge=0.2,
            le=1.0,
            default=0.5
        ),
        seed: int = Input(
            description="Random seed for reproducible results (0 for random generation)",
            default=0
        ),
    ) -> CogPath:
        """
        Generate high-quality speech audio from text using Chatterbox Multilingual model with optional reference audio styling.
        
        This tool synthesizes natural-sounding speech from input text. When a reference audio file 
        is provided, it captures the speaker's voice characteristics and speaking style. The generated audio 
        maintains the prosody, tone, and vocal qualities of the reference speaker, or uses default voice if no reference is provided.
        """
        
        # Input validation and processing
        if not text.strip():
            raise ValueError("Text input cannot be empty")

        # Set reproducible seed - following app.py pattern
        if seed != 0:
            set_seed(seed, self.device)

        # Truncate text to maximum characters (following app.py)
        text_to_synthesize = text[:300]
        if len(text) > 300:
            print(f"⚠️  Text truncated from {len(text)} to 300 characters")

        print(f"🗣️  Generating audio for text: '{text_to_synthesize[:50]}{'...' if len(text_to_synthesize) > 50 else ''}'")
        print(f"🌍 Language: {SUPPORTED_LANGUAGES[language]} ({language})")

        # Handle optional audio prompt - following app.py resolve_audio_prompt pattern
        audio_prompt_path = self._resolve_audio_prompt(language, reference_audio)
        
        # Prepare generation kwargs - following app.py structure
        generate_kwargs = {
            "exaggeration": exaggeration,
            "temperature": temperature,
            "cfg_weight": cfg_weight,
        }
        
        if audio_prompt_path:
            generate_kwargs["audio_prompt_path"] = audio_prompt_path
            print(f"📎 Using audio prompt: {audio_prompt_path}")
        else:
            print("ℹ️  No audio prompt provided; using default voice.")

        # Generate speech - following app.py generate call pattern
        wav = self.model.generate(
            text_to_synthesize,
            language_id=language,
            **generate_kwargs
        )
        
        print("✅ Audio generation complete.")
        
        # Convert to numpy and save output - following app.py return pattern
        audio_array = wav.squeeze(0).cpu().numpy()
        output_path = Path("/tmp/output.wav")
        wavfile.write(str(output_path), self.model.sr, audio_array)
        
        # Clean up memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return CogPath(output_path)

    def _resolve_audio_prompt(self, language_id: str, provided_audio: Optional[CogPath]) -> Optional[str]:
        """
        Decide which audio prompt to use - following app.py resolve_audio_prompt pattern:
        - If user provided a path (upload), use it.
        - Else, fall back to language-specific default (if any).
        """
        if provided_audio:
            return str(provided_audio)
        
        # Language-specific default voices - matches app.py LANGUAGE_CONFIG
        default_voice_urls = {
            "ar": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_m1.flac",
            "da": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
            "de": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
            "el": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
            "en": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
            "es": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
            "fi": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
            "fr": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
            "he": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
            "hi": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
            "it": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
            "ja": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja_f.flac",
            "ko": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
            "ms": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
            "nl": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
            "no": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
            "pl": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
            "pt": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
            "ru": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
            "sv": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
            "sw": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
            "tr": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
            "zh": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f.flac",
        }
        
        url = default_voice_urls.get(language_id)
        if not url:
            return None
            
        return self._download_reference_audio(url)

    def _download_reference_audio(self, url: str) -> Optional[str]:
        """Download reference audio with proper error handling"""
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            temp_path = tempfile.mktemp(suffix=".flac")
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"🎵 Downloaded default reference voice for language")
            return temp_path
            
        except requests.RequestException as e:
            print(f"⚠️  Failed to download reference voice: {e}")
            print("ℹ️  Continuing without reference audio")
            return None
