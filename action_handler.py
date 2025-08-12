#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Action Handler for the Neurofeedback System.

This script runs as a dedicated, background service responsible for playing
all adaptive audio feedback. It listens for commands over a UDP network socket
sent by the main 'block_meditation_analyzer.py' script.

Its sole purpose is to receive a command (e.g., "PLAY:Meditative:3") and
play the corresponding audio file, ensuring that audio playback does not
interfere with the main analysis loop. It includes several fallback methods
for audio playback to improve cross-platform compatibility.
"""

import socket
import os
import time
import random
import argparse
import traceback
import threading
import sys

# --- Audio Backend Selection ---
# Attempts to import preferred audio libraries first, with fallbacks for basic support.
try:
    import simpleaudio as sa
    AUDIO_BACKEND = "simpleaudio"
except ImportError:
    try:
        import sounddevice as sd
        import soundfile as sf
        AUDIO_BACKEND = "sounddevice"
    except ImportError:
        # If libraries are not installed, fall back to basic OS commands.
        if sys.platform == "darwin":  # macOS
            AUDIO_BACKEND = "afplay"
        elif sys.platform == "win32": # Windows
            AUDIO_BACKEND = "windows"
        else:  # Linux
            AUDIO_BACKEND = "aplay"

print(f"Using '{AUDIO_BACKEND}' for audio playback.")

# --- Default Configuration ---
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 12345
DEFAULT_AUDIO_PATH = "audio_manuscript"
DEFAULT_VOICE = "human"
NUM_VARIATIONS = {
    "High Variance": 6, "Meditative": 6, "Relaxed": 6, "Sleepy": 6
}

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Action Handler for Meditation Feedback")
parser.add_argument('--host', type=str, default=DEFAULT_HOST, help='Host IP to bind the UDP server to.')
parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='Port for the UDP server to listen on.')
parser.add_argument('--audio_path', type=str, default=DEFAULT_AUDIO_PATH, help='Path to the audio prompts folder.')
parser.add_argument('--voice', type=str, default=DEFAULT_VOICE, choices=['human', 'robot'], help='Voice type label.')
parser.add_argument('--debug', action='store_true', help='Enable debug output.')
args = parser.parse_args()

# --- Global Variables ---
HOST = args.host
PORT = args.port
AUDIO_PATH = args.audio_path
VOICE_TYPE = args.voice
DEBUG = args.debug

# A lock is used to prevent multiple audio files from playing at the same time.
audio_playing = False
audio_lock = threading.Lock()
stop_event = threading.Event()

def debug_print(*d_args, **d_kwargs):
    """A helper function to print messages only when DEBUG is True."""
    if DEBUG: print(*d_args, **d_kwargs)

# --- Audio Playback Functions ---
def _play_with_simpleaudio(filepath):
    """Helper to play WAV files using the simpleaudio library."""
    global audio_playing
    try:
        wave_obj = sa.WaveObject.from_wave_file(filepath)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"simpleaudio playback error: {e}")
    finally:
        with audio_lock:
            audio_playing = False

def _play_with_sounddevice(filepath):
    """Helper to play audio files using sounddevice and soundfile."""
    global audio_playing
    try:
        data, fs = sf.read(filepath, dtype='float32')
        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        print(f"sounddevice playback error: {e}")
    finally:
        with audio_lock:
            audio_playing = False

def _play_with_system(filepath):
    """Helper to play audio using native OS commands as a fallback."""
    global audio_playing
    command = ""
    try:
        if AUDIO_BACKEND == "afplay":    # macOS
            command = f"afplay \"{filepath}\""
        elif AUDIO_BACKEND == "aplay":   # Linux
            command = f"aplay \"{filepath}\""
        elif AUDIO_BACKEND == "windows": # Windows
            command = f"start /min wmplayer \"{filepath}\""
        
        if command:
            os.system(command)

    except Exception as e:
        print(f"System audio playback error: {e}")
    finally:
        with audio_lock:
            audio_playing = False

def play_audio(filepath):
    """
    Triggers audio playback in a new thread if no other audio is currently playing.

    This function acts as the main entry point for playing a sound. It checks the
    audio lock and, if available, dispatches the appropriate playback helper
    function to a separate thread so the main server loop doesn't block.
    """
    global audio_playing
    if not os.path.exists(filepath):
        print(f"Error: Audio file not found - {filepath}")
        return

    with audio_lock:
        # If audio is already playing, skip this request to avoid overlapping sounds.
        if audio_playing:
            debug_print(f"Audio skipped (already playing): {os.path.basename(filepath)}")
            return
        audio_playing = True
    
    # Select the correct playback function based on the available backend.
    playback_function = None
    if AUDIO_BACKEND == "simpleaudio":
        playback_function = _play_with_simpleaudio
    elif AUDIO_BACKEND == "sounddevice":
        playback_function = _play_with_sounddevice
    else: # afplay, aplay, windows
        playback_function = _play_with_system
    
    # Run the selected playback function in a non-blocking thread.
    threading.Thread(target=playback_function, args=(filepath,), daemon=True).start()

# --- UDP Server Class ---
class UDPServer:
    """A simple wrapper class for the UDP socket server."""
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None

    def start(self):
        """Binds the socket and prepares it to receive data."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # This option allows the script to be restarted quickly without waiting for the OS to release the port.
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            print(f"Action Handler listening on UDP {self.host}:{self.port}...")
            return True
        except socket.error as e:
            print(f"FATAL: Socket error on startup: {e}")
            if e.errno == 48: # Address already in use
                print("Hint: Another process is using this port. Try a different one using the --port argument.")
            return False

    def receive(self):
        """Blocks until a UDP packet is received and returns its content."""
        try:
            data, addr = self.socket.recvfrom(1024) # Buffer size 1024 bytes
            return data.decode('utf-8'), addr
        except socket.error as e:
            print(f"Error receiving data: {e}")
            return None, None
            
    def stop(self):
        """Closes the server socket."""
        if self.socket:
            self.socket.close()
            print("UDP server socket closed.")

# --- Main Server Loop ---
def run_server():
    """Initializes and runs the main UDP server loop."""
    server = UDPServer(HOST, PORT)
    if not server.start():
        print("Failed to start server. Exiting.")
        return
    
    print(f"Selected voice type: '{VOICE_TYPE}' | Audio path: '{AUDIO_PATH}'")
    if not os.path.isdir(AUDIO_PATH):
        print(f"Warning: Audio directory not found at '{AUDIO_PATH}'!")
    
    try:
        while not stop_event.is_set():
            command_str, addr = server.receive()
            if not command_str:
                continue
                
            debug_print(f"Received command: '{command_str}' from {addr}")
            
            # The expected command format is "PLAY:State Name:VariationNumber"
            if command_str.startswith("PLAY:"):
                parts = command_str.split(':', 2)
                if len(parts) == 3:
                    _, state_name, variation_str = parts
                    try:
                        # Construct the filename from the command parts.
                        state_name_for_file = state_name.lower().replace(' ', '_')
                        filename = f"{state_name_for_file}_{variation_str}.wav"
                        filepath = os.path.join(AUDIO_PATH, VOICE_TYPE, filename)
                        
                        play_audio(filepath)
                    except ValueError:
                        print(f"Warning: Invalid variation number in command: '{command_str}'")
                else:
                    print(f"Warning: Malformed PLAY command received: '{command_str}'")
                    
            elif command_str.strip().upper() == "EXIT":
                print("Received EXIT command. Shutting down.")
                break
                
            else:
                print(f"Warning: Unknown command received: '{command_str}'")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Shutting down.")
    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        traceback.print_exc()
    finally:
        print("Closing Action Handler.")
        stop_event.set()
        server.stop()

if __name__ == "__main__":
    run_server()