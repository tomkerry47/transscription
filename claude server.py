#!/usr/bin/env python3
"""
Real-time Whisper WebSocket Server for THRIVE Assessment Tool
Supports continuous audio streaming and transcription
FINAL VERSION - Fixed WebSocket handler signature
"""

import asyncio
import json
import logging
import numpy as np
import torch
import whisper
import websockets
from websockets.exceptions import ConnectionClosed
import threading
import time
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self, model_name="medium", device=None):
        """Initialize Whisper model"""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading Whisper model '{model_name}' on device: {device}")
        self.model = whisper.load_model(model_name, device=device)
        self.device = device
        
        # Audio parameters (must match client)
        self.sample_rate = 16000
        self.chunk_duration = 2.0  # Process 2-second chunks
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        # Buffer for incoming audio
        self.audio_buffer = deque()
        self.buffer_lock = threading.Lock()
        
        logger.info("Whisper transcriber initialized successfully")
    
    def add_audio_data(self, audio_bytes):
        """Add raw audio data to buffer"""
        try:
            # Convert bytes to numpy array (16-bit PCM)
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            with self.buffer_lock:
                self.audio_buffer.extend(audio_data)
                
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
    
    def get_audio_chunk(self):
        """Get a chunk of audio for processing"""
        with self.buffer_lock:
            if len(self.audio_buffer) >= self.chunk_samples:
                chunk = np.array([self.audio_buffer.popleft() for _ in range(self.chunk_samples)])
                return chunk
        return None
    
    def transcribe_chunk(self, audio_chunk):
        """Transcribe an audio chunk"""
        try:
            if len(audio_chunk) == 0:
                return None
                
            # Pad or trim to expected length
            if len(audio_chunk) < self.chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, self.chunk_samples - len(audio_chunk)))
            elif len(audio_chunk) > self.chunk_samples:
                audio_chunk = audio_chunk[:self.chunk_samples]
            
            # Transcribe using Whisper
            result = self.model.transcribe(audio_chunk, fp16=torch.cuda.is_available())
            
            text = result.get("text", "").strip()
            if text:
                logger.info(f"Transcribed: {text}")
                return {
                    "text": text,
                    "is_final": True,
                    "confidence": 1.0  # Whisper doesn't provide confidence scores
                }
        except Exception as e:
            logger.error(f"Transcription error: {e}")
        
        return None

class WebSocketServer:
    def __init__(self, host="0.0.0.0", port=8000, model_name="medium"):
        self.host = host
        self.port = port
        self.transcriber = WhisperTranscriber(model_name=model_name)
        self.active_connections = set()
        
    async def handler(self, websocket, path):
        """Handle WebSocket client connection - using handler name to match websockets library"""
        # Check if the client is connecting to the correct path
        if path != "/ws/transcribe":
            logger.warning(f"Client attempted connection to invalid path: {path}")
            await websocket.close(code=1008, reason="Invalid path")
            return
            
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_id} on path: {path}")
        self.active_connections.add(websocket)
        
        # Start transcription task for this client
        transcription_task = asyncio.create_task(
            self.transcription_loop(websocket, client_id)
        )
        
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Raw audio data
                    logger.debug(f"Received {len(message)} bytes of audio from {client_id}")
                    self.transcriber.add_audio_data(message)
                else:
                    # Text message (control commands)
                    try:
                        data = json.loads(message)
                        logger.info(f"Received control message from {client_id}: {data}")
                        
                        # Handle specific control messages
                        if data.get("type") == "start":
                            logger.info(f"Starting transcription for {client_id}")
                        elif data.get("type") == "stop":
                            logger.info(f"Stopping transcription for {client_id}")
                            break
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from {client_id}: {message}")
                        
        except ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self.active_connections.discard(websocket)
            transcription_task.cancel()
            logger.info(f"Cleaned up connection for {client_id}")
    
    async def transcription_loop(self, websocket, client_id):
        """Continuous transcription loop for a client"""
        try:
            while websocket in self.active_connections:
                # Get audio chunk from buffer
                chunk = self.transcriber.get_audio_chunk()
                if chunk is not None:
                    # Transcribe the chunk
                    result = self.transcriber.transcribe_chunk(chunk)
                    if result:
                        # Send result to client
                        message = json.dumps(result)
                        await websocket.send(message)
                        logger.debug(f"Sent transcription to {client_id}: {result['text'][:50]}...")
                
                # Small delay to prevent overwhelming the CPU
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            logger.info(f"Transcription loop cancelled for {client_id}")
        except Exception as e:
            logger.error(f"Transcription loop error for {client_id}: {e}")
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"Starting Whisper WebSocket server on {self.host}:{self.port}")
        
        # Create server with proper handler
        server = await websockets.serve(
            self.handler,  # Use the handler method
            self.host,
            self.port
        )
        
        logger.info(f"✅ Whisper server listening at ws://{self.host}:{self.port}/ws/transcribe")
        logger.info("✅ Server ready to accept connections from THRIVE app")
        logger.info(f"✅ Using model: {self.transcriber.model.device}")
        
        return server

async def health_check(websocket, path):
    """Simple health check endpoint"""
    if path == "/health":
        await websocket.send(json.dumps({"status": "healthy", "service": "whisper-server"}))
        await websocket.close()
    else:
        await websocket.close(code=1008, reason="Invalid path")

def main():
    """Main function to start the server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Whisper WebSocket Server for THRIVE")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", default="medium", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="Whisper model to use")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"✅ CUDA version: {torch.version.cuda}")
    else:
        logger.warning("⚠️  CUDA not available, using CPU (will be slower)")
    
    # Create and start server
    server = WebSocketServer(host=args.host, port=args.port, model_name=args.model)
    
    async def run_server():
        """Run the server with proper error handling"""
        try:
            server_instance = await server.start_server()
            logger.info("🚀 Server started successfully! Press Ctrl+C to stop.")
            
            # Keep the server running indefinitely
            await server_instance.wait_closed()
            
        except Exception as e:
            logger.error(f"❌ Server error: {e}")
            raise
    
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("🛑 Server shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Fatal server error: {e}")

if __name__ == "__main__":
    main()
