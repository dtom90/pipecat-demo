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
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

load_dotenv(override=True)

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
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        interim_results=False,  # Only get final results
        endpointing=True,       # Enable automatic speech end detection
    )

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY")
    )

    # Create a pipeline with the correct sequence of processors
    pipeline = Pipeline([
        transport.input(),
        stt,
        EchoProcessor(), # Add the echo processor here
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
