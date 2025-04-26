#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import string

from dotenv import load_dotenv
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.services.google.llm import GoogleLLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import LiveOptions
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

load_dotenv(override=True)

DAILY_API_KEY = os.getenv("DAILY_API_KEY")
DAILY_ROOM_URL = os.getenv("DAILY_ROOM_URL")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CONVERSATION_MODEL = "gemini-2.0-flash"
conversation_system_instruction = """
Pretend that you are a student applying to colleges. You want to get into the best college for you. You are asking for advice from a professional college advisor.
Keep your questions brief, concise, and to the point. Don't ask too many questions.
Your output will be converted to audio so don't include special characters in your answers.
"""

bot_name = "Student Bot"

async def run_bot():
    logger.info(f"Starting Daily.co bot")

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
        api_key=DEEPGRAM_API_KEY
    )

    conversation_llm = GoogleLLMService(
        name="Conversation",
        model=CONVERSATION_MODEL,
        api_key=GEMINI_API_KEY,
        system_instruction=conversation_system_instruction,
    )

    context = OpenAILLMContext()
    context_aggregator = conversation_llm.create_context_aggregator(context)

    # Initialize the text-to-speech service
    tts = DeepgramTTSService(
        api_key=DEEPGRAM_API_KEY,
        voice="aura-2-apollo-en"
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
        # await task.queue_frames([
        #     TTSSpeakFrame(f"{greeting}"),
        #     # No EndFrame here to keep the pipeline running
        # ])

    # Create and run the pipeline
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


if __name__ == "__main__":
    from run import main
    
    main()