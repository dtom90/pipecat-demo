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
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame, TTSSpeakFrame, EndFrame, TranscriptionFrame

load_dotenv(override=True)

DAILY_API_KEY = os.getenv("DAILY_API_KEY")
DAILY_ROOM_URL = os.getenv("DAILY_ROOM_URL")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

greeting = "Hello there! My name is EchoBot. I'm going to repeat everything you say. If you get tired of me, just say 'goodbye' and I'll go away."
farewell = "It was a pleasure repeating your utterances. Goodbye!"

# Define a custom processor to echo transcriptions
class EchoProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction) # Ensure base processing happens

        if isinstance(frame, TranscriptionFrame):
            if frame.text and frame.text.strip():
                # Prepare text: remove punctuation, strip whitespace, lowercase
                processed_text = frame.text.translate(str.maketrans('', '', string.punctuation)).strip().lower()
                logger.info(f"EchoProcessor recognized: {frame.text}, Processed: {processed_text}")

                # Check for goodbye after processing
                if processed_text == "goodbye":
                    logger.info("Goodbye detected, ending session.")
                    await self.push_frame(TTSSpeakFrame(farewell))
                    await self.push_frame(EndFrame())
                    return

                # Create a TTSSpeakFrame with the original transcribed text (as reverted by user)
                await self.push_frame(TTSSpeakFrame(f"You said: {frame.text}"))
            # We consume the TranscriptionFrame here
        else:
            # Let other frames pass through
            await self.push_frame(frame, direction)

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
            audio_in_enabled=True,
        ),
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        interim_results=False,  # Only get final results
        endpointing=True,       # Enable automatic speech end detection
    )

    # Initialize the text-to-speech service
    tts = DeepgramTTSService(
        api_key=DEEPGRAM_API_KEY
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        EchoProcessor(), # Add the echo processor here
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