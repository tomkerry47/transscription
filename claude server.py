#!/usr/bin/env python3
"""
Real-time Whisper WebSocket Server for THRIVE Assessment Tool
VERSION 2 - Enhanced with longer phrases and large-v3-turbo support
Supports continuous audio streaming and transcription with improved sentence formation
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
        """Initialize Whisper model with validation"""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validate model name
        if not self._validate_model(model_name):
            logger.warning(f"Model '{model_name}' may not be available. Falling back to 'medium'.")
            model_name = "medium"
        
        logger.info(f"Loading Whisper model '{model_name}' on device: {device}")
        self.model = whisper.load_model(model_name, device=device)
        self.device = device
        
        # Audio parameters (must match client - updated for longer phrases)
        self.sample_rate = 16000
        self.chunk_duration = 4.0  # Process 4-second chunks for better sentences
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        # Sentence handling for better phrase formation
        self.sentence_buffer = ""
        self.min_words_for_sentence = 3
        
        # Buffer for incoming audio
        self.audio_buffer = deque()
        self.buffer_lock = threading.Lock()
        
        logger.info(f"Whisper transcriber initialized successfully with {self.chunk_duration}s chunks")
    
    def _validate_model(self, model_name):
        """Validate if the model name is supported"""
        valid_models = [
            "tiny", "base", "small", "medium", "large", 
            "large-v1", "large-v2", "large-v3", "large-v3-turbo"
        ]
        return model_name in valid_models
    
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
        """Transcribe an audio chunk with improved sentence handling"""
        try:
            if len(audio_chunk) == 0:
                return None
                
            # Pad or trim to expected length
            if len(audio_chunk) < self.chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, self.chunk_samples - len(audio_chunk)))
            elif len(audio_chunk) > self.chunk_samples:
                audio_chunk = audio_chunk[:self.chunk_samples]
            
            # Transcribe using Whisper with optimized parameters
            result = self.model.transcribe(
                audio_chunk, 
                fp16=torch.cuda.is_available(),
                language="en",  # Force English for better performance
                word_timestamps=False,  # Disable for speed
                condition_on_previous_text=True  # Better continuity
            )
            
            text = result.get("text", "").strip()
            if text:
                # Improve sentence formation
                processed_text = self._process_text_for_sentences(text)
                if processed_text:
                    logger.info(f"Transcribed: {processed_text}")
                    return {
                        "text": processed_text,
                        "is_final": True,
                        "confidence": 1.0
                    }
        except Exception as e:
            logger.error(f"Transcription error: {e}")
        
        return None
    
    def _process_text_for_sentences(self, text):
        """Process text to form better sentences"""
        # Add text to sentence buffer
        self.sentence_buffer += " " + text
        self.sentence_buffer = self.sentence_buffer.strip()
        
        # Check if we have enough words for a meaningful phrase
        words = self.sentence_buffer.split()
        if len(words) >= self.min_words_for_sentence:
            # Look for natural sentence endings
            sentence_endings = ['.', '!', '?', ',']
            for ending in sentence_endings:
                if ending in self.sentence_buffer:
                    # Split at the last sentence ending
                    parts = self.sentence_buffer.rsplit(ending, 1)
                    if len(parts) > 1:
                        complete_sentence = parts[0] + ending
                        self.sentence_buffer = parts[1].strip()
                        return complete_sentence.strip()
            
            # If no sentence ending found but we have enough words, return the buffer
            if len(words) >= 8:  # Longer phrases for better context
                result = self.sentence_buffer
                self.sentence_buffer = ""
                return result
        
        return None

class WebSocketServer:
    def __init__(self, host="0.0.0.0", port=8000, model_name="medium"):
        self.host = host
        self.port = port
        self.transcriber = WhisperTranscriber(model_name=model_name)
        self.active_connections = set()
        logger.info(f"WebSocket server initialized with {model_name} model")
    
    async def handle_client(self, websocket):
        """Handle WebSocket client connection - Fixed signature for websockets library"""
        # Get the path from the websocket request
        path = websocket.path if hasattr(websocket, 'path') else "/ws/transcribe"
        
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
        logger.info(f"Starting Whisper WebSocket server v2 on {self.host}:{self.port}")
        
        # Create server with proper handler
        server = await websockets.serve(
            self.handle_client,  # Use the handle_client method
            self.host,
            self.port
        )
        
        logger.info(f"✅ Whisper server v2 listening at ws://{self.host}:{self.port}/ws/transcribe")
        logger.info("✅ Server ready to accept connections from THRIVE app")
        logger.info(f"✅ Using model on device: {self.transcriber.device}")
        logger.info(f"✅ Processing {self.transcriber.chunk_duration}s chunks for better sentences")
        
        return server

def main():
    """Main function to start the server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Whisper WebSocket Server v2 for THRIVE")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", default="medium", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3", "large-v3-turbo"],
                       help="Whisper model to use")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"✅ CUDA version: {torch.version.cuda}")
    else:
        logger.warning("⚠️  CUDA not available, using CPU (will be slower)")
    
    # Show model information
    logger.info(f"🎯 Starting with model: {args.model}")
    if args.model == "large-v3-turbo":
        logger.info("🚀 Using large-v3-turbo for highest quality and speed!")
    elif args.model == "large-v3":
        logger.info("🎯 Using large-v3 for highest quality!")
    elif args.model == "medium":
        logger.info("⚖️ Using medium model for balanced performance!")
    
    # Create and start server
    server = WebSocketServer(host=args.host, port=args.port, model_name=args.model)
    
    async def run_server():
        """Run the server with proper error handling"""
        try:
            server_instance = await server.start_server()
            logger.info("🚀 Server v2 started successfully! Press Ctrl+C to stop.")
            
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
