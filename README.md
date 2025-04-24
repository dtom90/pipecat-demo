# Pipecat Demo

This is a demo of Pipecat, a framework for building WebRTC pipelines.

## Prerequisites

- Python 3.11
- Deepgram API Key
- Gemini API Key

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
```

3. Set up environment variables:
```bash
DEEPGRAM_API_KEY=your_deepgram_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Running the Demos

1. Say one thing:
```bash
python 01-say-one-thing.py
```

2. Echo bot:
```bash
python 02-echobot.py
```

3. LLM bot with Gemini:
```bash
python 03-llmbot.py
```
