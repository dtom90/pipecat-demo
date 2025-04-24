#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import string

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame, Frame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection


load_dotenv(override=True)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CONVERSATION_MODEL = "gemini-2.0-flash"
conversation_system_instruction = """
You are a helpful LLM in a WebRTC call. Your goals are to be helpful and brief in your responses. Respond with one or two sentences at most, unless you are asked to
respond at more length. Your output will be converted to audio so don't include special characters in your answers.
"""

greeting = "Hello there! My name is Gemini. What can I help you with today?"

async def run_bot(webrtc_connection: SmallWebRTCConnection):
    logger.info(f"Starting bot")

    # Create a transport using the WebRTC connection
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_out_enabled=True,
            audio_in_enabled=True,  # Enable microphone input
        ),
    )

    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        interim_results=False,  # Only get final results
        endpointing=True,       # Enable automatic speech end detection
    )

    conversation_llm = GoogleLLMService(
        name="Conversation",
        model=CONVERSATION_MODEL,
        api_key=GEMINI_API_KEY,
        system_instruction=conversation_system_instruction,
    )

    context = OpenAILLMContext()
    context_aggregator = conversation_llm.create_context_aggregator(context)

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
    
    # Register an event handler so we can play the audio when the client joins
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await task.queue_frames([
            TTSSpeakFrame(f"{greeting}")
            # No EndFrame here to keep the pipeline running
        ])

    runner = PipelineRunner(handle_sigint=True)

    await runner.run(task)


if __name__ == "__main__":
    from run import main

    main()
