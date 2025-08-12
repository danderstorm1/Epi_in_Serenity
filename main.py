#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main execution script for the Neurofeedback Meditation System.

This script acts as the master launcher. Its primary role is to configure the
session using command-line arguments and then start the two core components
of the system in the correct order:

1. The 'action_handler.py' script is launched as a background process to handle
   all audio feedback prompts.
2. The 'block_meditation_analyzer.py' is then initialized and run in the foreground
   to manage the experiment, perform EEG analysis, and send commands.
"""

import time
import argparse
import sys
import traceback
import subprocess
import os 

# Import the main analyzer class and its default configuration.
from block_meditation_analyzer import (
    BlockBasedMeditationAnalyzer,
    DEFAULT_BASELINE_BLOCKS,
    DEFAULT_MEDITATION_BLOCKS,
    DEFAULT_BLOCK_DURATION,
    DEBUG as ANALYZER_DEFAULT_DEBUG
)


if __name__ == "__main__":

    # --- Setup Command-Line Argument Parsing ---
    # Defines how the user can configure the session from the terminal.
    parser = argparse.ArgumentParser(description="Main launcher for the Neurofeedback Meditation System")
    parser.add_argument('--debug', action='store_true', default=ANALYZER_DEFAULT_DEBUG, help='Enable debug output for both main script and action handler.')
    parser.add_argument('--blocks', type=int, default=DEFAULT_BASELINE_BLOCKS, help=f'Number of baseline blocks (default: {DEFAULT_BASELINE_BLOCKS})')
    parser.add_argument('--meditation', type=int, default=DEFAULT_MEDITATION_BLOCKS, help=f'Number of meditation blocks (default: {DEFAULT_MEDITATION_BLOCKS})')
    parser.add_argument('--duration', type=int, default=DEFAULT_BLOCK_DURATION, help=f'Duration of each analysis block in seconds (default: {DEFAULT_BLOCK_DURATION})')
    parser.add_argument('--voice', type=str, default="human", choices=['human', 'robot'], help='Voice type for feedback prompts (default: human).')
    parser.add_argument('--audio_path', type=str, default="audio_manuscript", help='Path to the audio prompts folder.')
    parser.add_argument('--ipc_host', type=str, default="127.0.0.1", help='Host IP for the action handler UDP server.')
    parser.add_argument('--ipc_port', type=int, default=12345, help='Port for the action handler UDP server.')
    parser.add_argument('--action_script', type=str, default="action_handler.py", help='Name/Path of the action handler script.')

    args = parser.parse_args()
    DEBUG = args.debug

    # --- Print Session Configuration ---
    print("--- Starting Neurofeedback Meditation Session ---")
    print(f"  - Session: {args.blocks} baseline blocks + {args.meditation} meditation blocks")
    print(f"  - Block Duration: {args.duration} seconds")
    print(f"  - Debug Mode: {DEBUG}")
    print(f"  - Action Handler UDP Target: {args.ipc_host}:{args.ipc_port}")
    print("-------------------------------------------------")

    # --- Start Action Handler as a Background Process ---
    # The action_handler is run in a separate process so that audio playback
    # does not block or interfere with the main, time-sensitive analysis loop.
    action_process = None
    try:
        # Build the command to run the action handler script with the same python interpreter.
        cmd = [
            sys.executable,
            args.action_script,
            "--audio_path", args.audio_path,
            "--voice", args.voice,
            "--ipc_host", args.ipc_host,
            "--port", str(args.ipc_port)
        ]
        if args.debug:
            cmd.append("--debug")

        print(f"Attempting to start {args.action_script} in the background...")
        # Popen runs the command without waiting for it to finish.
        action_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(f"{args.action_script} started with PID: {action_process.pid}.")

        # Give the action handler a moment to initialize and bind its network socket.
        time.sleep(2.5)

        # Sanity check: see if the process terminated immediately after starting.
        if action_process.poll() is not None:
             print(f"Error: {args.action_script} terminated unexpectedly after starting.")
             action_process = None # Ensure it's None so it's not terminated later.

    except FileNotFoundError:
        print(f"Error: The script '{args.action_script}' was not found.")
        action_process = None
    except Exception as e:
        print(f"An unexpected error occurred while starting {args.action_script}: {e}")
        traceback.print_exc()
        action_process = None

    # --- Initialize and Run the Main Analyzer ---
    analyzer = None
    try:
        if action_process is None:
             print("\nWarning: Action handler process is not running. Adaptive audio feedback will not work.")

        # Pass the parsed arguments to the analyzer's constructor.
        analyzer = BlockBasedMeditationAnalyzer(
            baseline_blocks=args.blocks,
            meditation_blocks=args.meditation,
            block_duration=args.duration,
            voice_type=args.voice,
            audio_path=args.audio_path,
            debug_mode=args.debug,
            ipc_host=args.ipc_host,
            ipc_port=args.ipc_port
        )
        # This call starts the main experiment loop.
        analyzer.run()

    except Exception as e:
        print(f"\nAn unexpected error occurred in the main analyzer: {e}")
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        # This block ensures that the background action_handler process is
        # properly terminated when the main script finishes or crashes.
        print("\nMain script finishing. Cleaning up background processes...")
        if action_process and action_process.poll() is None:
            print(f"Terminating {args.action_script} (PID: {action_process.pid})...")
            action_process.terminate() # Primary attempt of stopping.
            try:
                action_process.wait(timeout=2) # Wait up to 2 seconds for it to close.
                print(f"{args.action_script} terminated.")
            except subprocess.TimeoutExpired:
                # If it doesn't close in time it's forced to stop.
                print(f"Warning: {args.action_script} did not terminate gracefully. Forcing shutdown...")
                action_process.kill()

    print("\n--- Script Finished ---")