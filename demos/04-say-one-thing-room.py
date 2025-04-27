#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.services.daily import DailyTransport, DailyParams

load_dotenv(override=True)

DAILY_API_KEY = os.getenv("DAILY_API_KEY")
DAILY_ROOM_URL = os.getenv("DAILY_ROOM_URL")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# The welcome message to speak when the first participant joins
welcome_message = "Welcome to the Daily.co room! I'm your virtual assistant bot."

async def run_bot():
    logger.info(f"Starting Daily.co bot")

    # Create a transport using the Daily connection
    transport = DailyTransport(
        room_url=DAILY_ROOM_URL,
        bot_name="Pipecat Daily Bot",
        token=None,
        params=DailyParams( 
            api_key=DAILY_API_KEY,
            audio_out_enabled=True,
        ),
    )

    # Initialize the text-to-speech service
    tts = DeepgramTTSService(
        api_key=DEEPGRAM_API_KEY
    )

    # Create a pipeline task with the TTS service and transport output
    task = PipelineTask(Pipeline([tts, transport.output()]))

    # Register an event handler for when a participant joins
    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        logger.info(f"Participant joined: {participant['id']}")
            
        # Queue the welcome message and end frame
        await task.queue_frames([
            TTSSpeakFrame(welcome_message),
            # EndFrame()
        ])

    # Create and run the pipeline
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


if __name__ == "__main__":
    from run import main
    
    main()