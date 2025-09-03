# Chatterbox Multilingual TTS

Turn text into natural speech in 23 languages. Clone voices too.

Built by ResembleAI, optimized for fast deployment.

## What this does

This model takes your text and turns it into realistic speech. Works in 23 languages: Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Swahili, Turkish.

Upload your own audio file and it'll try to clone that voice. Pretty cool.

## Why it's cool

- **Fast**: Downloads 3GB of model weights in 17 seconds (thanks to pget)
- **Multilingual**: Actually works well across 23 languages
- **Voice cloning**: Upload a voice sample, get speech in that voice
- **Just works**: No HuggingFace auth, no complicated setup

## How to run it

```bash
git clone https://github.com/zsxkib/cog-ResembleAI-Chatterbox-Multilingual-TTS
cd cog-ResembleAI-Chatterbox-Multilingual-TTS
cog predict -i text="Hello world" -i language="en"
```

That's it. Cog will build the container, download the weights, and generate speech.

Want to clone a voice? Add a reference audio file:

```bash
cog predict -i text="Hello in my voice" -i language="en" -i reference_audio=@voice.wav
```

Want more drama in the speech?

```bash
cog predict -i text="Very dramatic speech" -i language="en" -i exaggeration=0.8
```

## Requirements

- [Cog](https://github.com/replicate/cog)
- Docker
- That's it

The first run downloads model weights automatically. Subsequent runs are fast.

## License

MIT
