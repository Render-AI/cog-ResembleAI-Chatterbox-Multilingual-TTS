import os
import random
import tempfile
import requests
import gc
import torch.cuda

from typing import Optional
import numpy as np
import torch
import scipy.io.wavfile as wavfile
from cog import BasePredictor, Input, Path

# Fixed import path from app.py
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
# Optimized Chatterbox Multilingual TTS for Replicate Cog
# - GPU memory optimizations for A100 and lower-end cards
# - Efficient audio prompt downloading with streaming
# - Memory cleanup after generation
# - TF32 enabled for better performance on modern GPUs


# Language configuration copied from app.py
LANGUAGE_CONFIG = {
    "ar": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_m1.flac",
        "text": "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب."
    },
    "da": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
        "text": "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal."
    },
    "de": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
        "text": "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal."
    },
    "el": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
        "text": "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube."
    },
    "en": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel."
    },
    "es": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
        "text": "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube."
    },
    "fi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
        "text": "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme."
    },
    "fr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
        "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube."
    },
    "he": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
        "text": "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו."
    },
    "hi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
        "text": "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।"
    },
    "it": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
        "text": "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube."
    },
    "ja": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja_f.flac",
        "text": "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。"
    },
    "ko": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
        "text": "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다."
    },
    "ms": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
        "text": "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami."
    },
    "nl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
        "text": "Vorige maand bereikten we een nieuwe mijlpaal met twee miljard weergaven op ons YouTube-kanaal."
    },
    "no": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
        "text": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår."
    },
    "pl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
        "text": "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube."
    },
    "pt": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
        "text": "No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube."
    },
    "ru": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
        "text": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале."
    },
    "sv": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
        "text": "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal."
    },
    "sw": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
        "text": "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kweny kituo chetu cha YouTube."
    },
    "tr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
        "text": "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık."
    },
    "zh": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f.flac",
        "text": "上个月，我们达到了一个新的里程碑，我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。"
    },
}


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model - adapted from app.py get_or_load_model()"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Running on device: {self.device}")
        
        try:
            self.model = ChatterboxMultilingualTTS.from_pretrained(self.device)
            if hasattr(self.model, 'to') and str(self.model.device) != self.device:
                self.model.to(self.device)
            print(f"Model loaded successfully. Device: {getattr(self.model, 'device', 'N/A')}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Memory optimization for better GPU utilization
        if self.device == "cuda":
            torch.cuda.empty_cache()
            # Enable memory efficient attention
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Enable optimized memory format
            torch.backends.cudnn.benchmark = True

    def set_seed(self, seed: int):
        """Set random seed - copied from app.py"""
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)

    def default_audio_for_ui(self, lang: str) -> str:
        """Get default audio prompt for language - with URL download support"""
        url = LANGUAGE_CONFIG.get(lang, {}).get("audio")
        if url and url.startswith('http'):
            # Download the URL to a temporary file
            try:
                print(f"Downloading audio prompt from {url}")
                response = requests.get(url, timeout=30, stream=True)
                response.raise_for_status()
                temp_path = tempfile.mktemp(suffix=".flac")
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return temp_path
            except Exception as e:
                print(f"Warning: Failed to download audio prompt from {url}: {e}")
                print("Continuing without reference audio...")
                return None
        return url

    def predict(
        self,
        text_to_synthesize: str = Input(description="Text to synthesize into speech (max 300 chars)"),
        language_id: str = Input(
            description="Language code",
            choices=list(LANGUAGE_CONFIG.keys()),
            default="en"
        ),
        reference_audio: Optional[Path] = Input(
            description="Optional reference audio file for voice style",
            default=None
        ),
        exaggeration: float = Input(
            description="Speech expressiveness (0.25-2.0)",
            ge=0.25,
            le=2.0,
            default=0.5
        ),
        temperature: float = Input(
            description="Generation randomness (0.05-5.0)",
            ge=0.05,
            le=5.0,
            default=0.8
        ),
        seed: int = Input(
            description="Random seed (0 for random)",
            ge=0,
            default=0
        ),
        cfg_weight: float = Input(
            description="CFG weight (0.0-1.0)",
            ge=0.0,
            le=1.0,
            default=0.5
        )
    ) -> Path:
        """Generate TTS audio - logic copied from app.py generate_tts_audio()"""
        
        if not self.model:
            raise RuntimeError("TTS model is not loaded.")

        # Set seed if provided
        if seed != 0:
            self.set_seed(seed)

        print(f"Generating audio for text: '{text_to_synthesize[:50]}...'")
        
        # Handle optional audio prompt - logic from app.py
        if reference_audio:
            chosen_prompt = str(reference_audio)
        else:
            chosen_prompt = self.default_audio_for_ui(language_id)

        # Prepare generation kwargs - copied from app.py
        generate_kwargs = {
            "exaggeration": exaggeration,
            "temperature": temperature,
            "cfg_weight": cfg_weight,
        }
        
        if chosen_prompt:
            generate_kwargs["audio_prompt_path"] = chosen_prompt
            print(f"Using audio prompt: {chosen_prompt}")
        else:
            print("No audio prompt provided; using default voice.")

        try:
            # Core generation call - exactly from app.py
            wav = self.model.generate(
                text_to_synthesize[:300],  # Truncate text to max chars
                language_id=language_id,
                **generate_kwargs
            )
            print("Audio generation complete.")
            
            # Convert to numpy and get sample rate - from app.py
            sample_rate = self.model.sr
            audio_array = wav.squeeze(0).numpy()
            
            # Save to temporary file for Cog to serve
            output_path = Path(tempfile.mktemp(suffix=".wav"))
            wavfile.write(str(output_path), sample_rate, audio_array)
            print(f"Audio saved to: {output_path}")
            
            
            # Memory cleanup for better performance
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            return output_path
            
        except Exception as e:
            print(f"Error during audio generation: {e}")
            raise
