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
from pipecat.services.google.llm import GoogleLLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import LiveOptions
from pipecat.transcriptions.language import Language
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.vad.silero import SileroVADAnalyzer


load_dotenv(override=True)

DAILY_API_KEY = os.getenv("DAILY_API_KEY")
DAILY_ROOM_URL = os.getenv("DAILY_ROOM_URL")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CONVERSATION_MODEL = "gemini-2.0-flash"
CONVERSATION_SYSTEM_INSTRUCTION = """
You are a professional college advisor. You are an expert in college admissions. Your goal is to help students apply to colleges and universities.
Your goals are to be helpful and brief in your responses.
Respond with one or two sentences at most, unless you are asked to respond at more length. 
Your output will be converted to audio so don't include special characters in your answers.
"""

BOT_NAME = "College Advisor Bot"
GREETING = f"Hello there! My name is {BOT_NAME}. How can I help you today?"

STOP_SECONDS = 1

async def run_bot():
    logger.info(f"Starting Daily.co bot")

    # Create a transport using the Daily connection
    transport = DailyTransport(
        room_url=DAILY_ROOM_URL,
        bot_name=BOT_NAME,
        token=None,
        params=DailyParams( 
            api_key=DAILY_API_KEY,
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_enabled=True,
            vad_audio_passthrough=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=STOP_SECONDS)),
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

    conversation_llm = GoogleLLMService(
        name=BOT_NAME,
        model=CONVERSATION_MODEL,
        api_key=GEMINI_API_KEY,
        system_instruction=CONVERSATION_SYSTEM_INSTRUCTION,
    )

    context = OpenAILLMContext()
    context_aggregator = conversation_llm.create_context_aggregator(context)

    # Initialize the text-to-speech service
    tts = DeepgramTTSService(
        api_key=DEEPGRAM_API_KEY
    )

    # Create a pipeline with the correct sequence of processors
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        conversation_llm,
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
            TTSSpeakFrame(f"{GREETING}"),
            # No EndFrame here to keep the pipeline running
        ])

    # Create and run the pipeline
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


if __name__ == "__main__":
    from run import main
    
    main()
