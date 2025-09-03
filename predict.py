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
import random
import tempfile
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


class Predictor(BasePredictor):
    """Chatterbox Multilingual TTS Predictor for Replicate"""

    def setup(self) -> None:
        """
        Load the multilingual TTS model
        
        Note: This model requires pre-downloaded weights from a private HuggingFace repository.
        To download the model weights, use:
        
        huggingface-cli download ResembleAI/Chatterbox-Multilingual-AllLang \
            --local-dir model_cache \
            --token YOUR_HF_TOKEN
            
        The model files should be placed in the model_cache/ directory:
        - t3_alllang.safetensors (2.0GB) - Main multilingual TTS model
        - s3gen.pt (1.0GB) - Speech generator model  
        - ve.pt (5.5MB) - Voice encoder for speaker embeddings
        - conds.pt (105KB) - Default voice conditionals
        - mtl_tokenizer.json (67KB) - Multilingual tokenizer
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing Chatterbox Multilingual TTS on {self.device}")
        
        # Optimize PyTorch settings for better performance
        if self.device == "cuda":
            # Enable TF32 for faster training/inference on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        
        # Load model with optimized settings
        try:
            self.model = ChatterboxMultilingualTTS.from_pretrained(self.device)
            print(f"✅ Model loaded successfully")
            print(f"📋 Supporting {len(SUPPORTED_LANGUAGES)} languages")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def predict(
        self,
        text: str = Input(
            description="Text to synthesize into speech",
            max_length=300
        ),
        language: str = Input(
            description="Language for synthesis",
            choices=list(SUPPORTED_LANGUAGES.keys()),
            default="en"
        ),
        reference_audio: Optional[CogPath] = Input(
            description="Reference audio file for voice cloning (optional)",
            default=None
        ),
        temperature: float = Input(
            description="Controls speech variation. Higher = more expressive",
            ge=0.05,
            le=5.0,
            default=0.8
        ),
        exaggeration: float = Input(
            description="Speech expressiveness level. Higher = more dramatic",
            ge=0.25,
            le=2.0,
            default=0.5
        ),
        cfg_weight: float = Input(
            description="Classifier-free guidance. 0=natural, 1=guided",
            ge=0.0,
            le=1.0,
            default=0.5
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducible generation",
            default=None
        ),
    ) -> CogPath:
        """
        Generate multilingual speech from text
        
        Returns:
            Audio file containing synthesized speech
        """
        # Input validation
        if not text.strip():
            raise ValueError("Text input cannot be empty")
        
        original_length = len(text)
        if original_length > 300:
            text = text[:300]
            print(f"⚠️  Text truncated from {original_length} to 300 characters")

        # Set reproducible seed
        if seed is not None:
            self._set_seed(seed)

        print(f"🗣️  Synthesizing: '{self._truncate_for_display(text, 50)}'")
        print(f"🌍 Language: {SUPPORTED_LANGUAGES[language]} ({language})")

        # Get reference audio (uploaded or default)
        reference_path = self._get_reference_audio(language, reference_audio)
        
        # Generate speech with error handling
        try:
            audio_array, sample_rate = self._generate_speech(
                text=text,
                language=language,
                reference_path=reference_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
            
            # Save output
            output_path = Path("/tmp/output.wav")
            wavfile.write(str(output_path), sample_rate, audio_array)
            
            # Clean up memory
            self._cleanup_memory()
            
            print("✅ Speech synthesis completed")
            return CogPath(output_path)
            
        except Exception as e:
            self._cleanup_memory()
            raise RuntimeError(f"Speech synthesis failed: {str(e)}")

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducible generation"""
        random.seed(seed)
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)

    def _truncate_for_display(self, text: str, max_len: int) -> str:
        """Truncate text for clean logging display"""
        return f"{text[:max_len]}..." if len(text) > max_len else text

    def _get_reference_audio(self, language: str, uploaded_audio: Optional[CogPath]) -> Optional[str]:
        """Get reference audio path - either uploaded file or language default"""
        if uploaded_audio:
            print(f"📎 Using uploaded reference audio")
            return str(uploaded_audio)
        
        # Language-specific default voices
        default_urls = {
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
        
        url = default_urls.get(language)
        if not url:
            print(f"ℹ️  No default voice available for {language}")
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
            
            print(f"🎵 Downloaded reference voice")
            return temp_path
            
        except Exception as e:
            print(f"⚠️  Failed to download reference voice: {e}")
            print("ℹ️  Continuing with model's default voice")
            return None

    def _generate_speech(
        self, 
        text: str, 
        language: str, 
        reference_path: Optional[str],
        temperature: float,
        exaggeration: float,
        cfg_weight: float
    ) -> tuple[np.ndarray, int]:
        """Generate speech using the multilingual TTS model"""
        
        # Suppress some model internal warnings during generation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Reference mel length.*")
            warnings.filterwarnings("ignore", message=".*return_dict_in_generate.*")
            warnings.filterwarnings("ignore", message=".*past_key_values.*")
            warnings.filterwarnings("ignore", message=".*LlamaModel is using.*")
            
            generation_kwargs = {
                "temperature": temperature,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
            }
            
            if reference_path:
                generation_kwargs["audio_prompt_path"] = reference_path
                
            # Generate audio tensor
            audio_tensor = self.model.generate(
                text=text,
                language_id=language,
                **generation_kwargs
            )
        
        # Convert to numpy array and get sample rate
        audio_array = audio_tensor.squeeze(0).cpu().numpy()
        sample_rate = self.model.sr
        
        return audio_array, sample_rate

    def _cleanup_memory(self) -> None:
        """Clean up GPU memory for optimal performance"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
