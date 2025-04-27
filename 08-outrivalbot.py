#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
from dotenv import load_dotenv
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.services.deepgram.stt import LiveOptions
from pipecat.transcriptions.language import Language
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from rag_agent import RagAgentProcessor


load_dotenv(override=True)

DAILY_API_KEY = os.getenv("DAILY_API_KEY")
DAILY_ROOM_URL = os.getenv("DAILY_ROOM_URL")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLM_MODEL_PROVIDER = "google_genai"
LLM_MODEL = "gemini-2.0-flash"

EMBEDDINGS_MODEL = "text-embedding-004"
WEBPAGE_DOCUMENTS = (
    'https://outrival.com/',
    'https://outrival.com/company',
    'https://outrival.com/blog/ai-for-the-people'
)

TOPIC = '"Outrival", the AI Company'
ADDITIONAL_CONTEXT = "Outrival is a company that specializes in AI and machine learning."

bot_name = "Outrival Bot"
greeting = f"Hello there! My name is {bot_name}. How can I help you today?"

async def run_bot():
    logger.info(f"Starting {bot_name}")

    # Create a transport using the Daily connection
    vad_analyzer = SileroVADAnalyzer(params=VADParams())
    transport = DailyTransport(
        room_url=DAILY_ROOM_URL,
        bot_name=bot_name,
        token=None,
        params=DailyParams( 
            api_key=DAILY_API_KEY,
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_enabled=True,
            vad_audio_passthrough=True,
            vad_analyzer=vad_analyzer,
        ),
    )

    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        live_options=LiveOptions(
            encoding="linear16",
            language=Language.EN,
            model="nova-3-general",
            sample_rate=16000,
            channels=1,
            smart_format=True,
            punctuate=True,
            profanity_filter=True,
        )
    )

    rag_agent = RagAgentProcessor(
        topic=TOPIC,
        llm_model=LLM_MODEL,
        llm_model_provider=LLM_MODEL_PROVIDER,
        embeddings_model=EMBEDDINGS_MODEL,
        webpage_documents=WEBPAGE_DOCUMENTS,
        name=bot_name
    )

    # Initialize the text-to-speech service
    tts = DeepgramTTSService(
        api_key=DEEPGRAM_API_KEY
    )

    # Create a pipeline with the correct sequence of processors
    pipeline = Pipeline([
        transport.input(),
        stt,
        rag_agent,
        tts,
        transport.output()
    ])

    task = PipelineTask(pipeline)

    # Register an event handler for when a participant joins
    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        logger.info(f"Participant joined: {participant['id']}")

        # Queue the welcome message and end frame
        await task.queue_frames([
            TTSSpeakFrame(f"{greeting}"),
            # No EndFrame here to keep the pipeline running
        ])

    # Create and run the pipeline
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


if __name__ == "__main__":
    from run import main
    
    main()