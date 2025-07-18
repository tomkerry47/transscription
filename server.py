import asyncio
import websockets
import json
import torch
import numpy as np
from transformers import pipeline

# Setup the Whisper pipeline
# If you have a CUDA-compatible GPU, it will be used. Otherwise, it will use the CPU.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base.en", # You can choose other models like whisper-medium.en for better accuracy
    device=device
)

async def transcribe(websocket):
    print("Client connected.")
    try:
        async for message in websocket:
            # The message is expected to be raw audio data (bytes)
            if isinstance(message, bytes):
                # Convert the raw bytes to a NumPy array of 32-bit floats
                audio_np = np.frombuffer(message, dtype=np.float32)

                # Process the audio data, providing the raw bytes and the sample rate
                audio_input = {"raw": audio_np, "sampling_rate": 16000}
                result = pipe(audio_input)
                # Send back the transcription result
                await websocket.send(json.dumps({
                    "text": result["text"]
                }))
    except websockets.exceptions.ConnectionClosedOK:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")

async def main():
    # The 'async with' statement ensures the server is properly closed
    # when the program exits.
    async with websockets.serve(transcribe, "0.0.0.0", 8765):
        print("Whisper server started on ws://0.0.0.0:8765. Ready for connections.")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server shutting down.")
