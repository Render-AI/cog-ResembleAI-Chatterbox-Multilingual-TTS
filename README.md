# Chatterbox Multilingual TTS

High-quality text-to-speech in 23 languages with voice cloning. Built by ResembleAI.

[![Replicate](https://replicate.com/zsxkib/chatterbox-multilingual-tts/badge)](https://replicate.com/zsxkib/chatterbox-multilingual-tts) 

## What this does

This model turns text into natural-sounding speech in 23 different languages. You can either use the default voice for each language, or upload your own audio file to clone a specific voice.

## Languages

Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Swahili, Turkish.

## How to use it

### Basic text-to-speech

```python
import replicate

output = replicate.run(
    "zsxkib/chatterbox-multilingual-tts",
    input={
        "text": "Hello, this is a test of multilingual speech synthesis.",
        "language": "en"
    }
)
```

### Voice cloning with reference audio

```python
import replicate

output = replicate.run(
    "zsxkib/chatterbox-multilingual-tts",
    input={
        "text": "This will sound like the voice in my reference audio.",
        "language": "en",
        "reference_audio": open("voice_sample.wav", "rb")
    }
)
```

### Advanced options

```python
import replicate

output = replicate.run(
    "zsxkib/chatterbox-multilingual-tts",
    input={
        "text": "More expressive speech with custom settings.",
        "language": "en",
        "exaggeration": 0.7,      # More dramatic (0.25-2.0)
        "temperature": 1.2,       # More varied (0.05-5.0) 
        "cfg_weight": 0.8,        # More guided (0.2-1.0)
        "seed": 42                # Reproducible results
    }
)
```

## Parameters

- **text**: The text you want to turn into speech (max 300 characters)
- **language**: Two-letter language code (like "en" for English)
- **reference_audio**: Upload an audio file to clone that voice (optional)
- **exaggeration**: How expressive the speech is. 0.5 is neutral, higher is more dramatic
- **temperature**: Randomness in generation. Higher values give more varied speech
- **cfg_weight**: How much the model follows guidance. 0.5 is balanced, higher is more controlled
- **seed**: Set to the same number to get the same result every time

## Examples by language

**English**: "Last month, we reached a new milestone with two billion views on our YouTube channel."

**Spanish**: "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube."

**French**: "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube."

**German**: "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal."

**Chinese**: "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。"

## Tips for better results

- **Keep text under 300 characters** - longer text gets cut off
- **Use punctuation** - it helps with natural pauses and rhythm  
- **Match your reference audio language** - the model works best when the reference voice speaks the same language as your text
- **Reference audio should be 3-10 seconds** - clean speech without background noise works best
- **Try different exaggeration values** - 0.3 for calm speech, 0.8 for more energetic

## About voice cloning

When you upload reference audio, the model tries to match the voice characteristics like pitch, accent, and speaking style. It works better with clear audio of a single speaker. The model can't perfectly clone any voice, but it gets pretty close for most cases.

## Model details

This is ResembleAI's Chatterbox Multilingual model. It uses a neural text-to-speech system trained on speech data from 23 languages. The model can generate high-quality audio at 24kHz sample rate.

## License

MIT
