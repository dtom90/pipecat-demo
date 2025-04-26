#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import datetime
from dotenv import load_dotenv
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.frames.frames import TTSSpeakFrame, TranscriptionFrame
from pipecat.services.google.llm import GoogleLLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import LiveOptions
from pipecat.transcriptions.language import Language
from pipecat.processors.aggregators.gated import GatedAggregator
from pipecat.frames.frames import TextFrame, ControlFrame
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.vad.silero import SileroVADAnalyzer


load_dotenv(override=True)

DAILY_API_KEY = os.getenv("DAILY_API_KEY")
DAILY_ROOM_URL = os.getenv("DAILY_ROOM_URL")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CONVERSATION_MODEL = "gemini-2.0-flash"
conversation_system_instruction = """
You are a professional college advisor. You are an expert in college admissions. Your goal is to help students apply to colleges and universities.
Your goals are to be helpful and brief in your responses.
Respond with one or two sentences at most, unless you are asked to respond at more length. 
Your output will be converted to audio so don't include special characters in your answers.
"""

bot_name = "College Advisor Bot"
greeting = f"Hello there! My name is {bot_name}. How can I help you today?"

class SpeechStartedFrame(ControlFrame):
    pass

class UtteranceEndedFrame(ControlFrame):
    pass

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
            # To get UtteranceEnd, the following must be set:
            interim_results=True,
            utterance_end_ms="2000",
            vad_events=True,
            # Time in milliseconds of silence to wait for before finalizing speech
            # endpointing=2000,
        )
    )

    # @stt.event_handler("on_speech_started")
    # async def on_speech_started(stt, *args, **kwargs):
    #     logger.info(f"Speech started")
    #     await task.queue_frames([
    #         SpeechStartedFrame()
    #     ])

    # # Instantiate the GatedAggregator
    # gated_aggregator = GatedAggregator(
    #     gate_open_fn=lambda x: isinstance(x, SpeechStartedFrame),
    #     gate_close_fn=lambda x: isinstance(x, UtteranceEndedFrame),
    #     start_open=False
    # )

    # @stt.event_handler("on_utterance_end")
    # async def on_utterance_end(stt, *args, **kwargs):
    #     logger.info(f"Utterance ended")
    #     await task.queue_frames([
    #         UtteranceEndedFrame()
    #     ])
        
    
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
        api_key=DEEPGRAM_API_KEY
    )

    # Create a pipeline with the correct sequence of processors
    pipeline = Pipeline([
        transport.input(),
        stt,
        # gated_aggregator, # Use the gated aggregator instance
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
            TTSSpeakFrame(f"{greeting}"),
            # No EndFrame here to keep the pipeline running
        ])

    # Create and run the pipeline
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


if __name__ == "__main__":
    from run import main
    
    main()