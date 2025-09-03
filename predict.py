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
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from scipy.io import wavfile
from cog import BasePredictor, Input, Path as CogPath

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES


class Predictor(BasePredictor):
    """Chatterbox Multilingual TTS Predictor for Replicate"""

    def setup(self) -> None:
        """Load the multilingual TTS model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing Chatterbox Multilingual TTS on {self.device}")
        
        # Load model from local files (downloaded during container build)
        self.model = ChatterboxMultilingualTTS.from_pretrained(self.device)
        
        # Optimize GPU performance
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        print(f"✅ Model loaded successfully on {self.device}")
        print(f"📋 Supports {len(SUPPORTED_LANGUAGES)} languages: {', '.join(SUPPORTED_LANGUAGES.keys())}")

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
        if len(text.strip()) == 0:
            raise ValueError("Text input cannot be empty")
        
        if len(text) > 300:
            text = text[:300]
            print(f"⚠️  Text truncated to 300 characters")

        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed_all(seed)

        print(f"🗣️  Synthesizing: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"🌍 Language: {SUPPORTED_LANGUAGES[language]} ({language})")

        # Resolve reference audio (uploaded file or default for language)
        reference_path = self._get_reference_audio(language, reference_audio)
        
        # Generate speech
        try:
            audio_array, sample_rate = self._generate_speech(
                text=text,
                language=language,
                reference_path=reference_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
            
            # Save to output file
            output_path = Path("/tmp/output.wav")
            wavfile.write(str(output_path), sample_rate, audio_array)
            
            # Memory cleanup for better GPU utilization
            self._cleanup_memory()
            
            print("✅ Speech synthesis completed successfully")
            return CogPath(output_path)
            
        except Exception as e:
            self._cleanup_memory()
            raise RuntimeError(f"Speech synthesis failed: {str(e)}")

    def _get_reference_audio(self, language: str, uploaded_audio: Optional[CogPath]) -> Optional[str]:
        """Get reference audio path - either uploaded file or language default"""
        if uploaded_audio:
            return str(uploaded_audio)
        
        # Use language-specific default voice
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
            return None
            
        # Download reference audio
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            temp_path = tempfile.mktemp(suffix=".flac")
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            print(f"📥 Downloaded reference voice for {language}")
            return temp_path
            
        except Exception as e:
            print(f"⚠️  Could not download reference audio: {e}")
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
        # Configure generation parameters
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

    def _cleanup_memory(self):
        """Clean up GPU memory for optimal performance"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
