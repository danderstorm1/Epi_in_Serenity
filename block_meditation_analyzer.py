#!/usr/bin/env python
# -*- coding: utf-8 -*-

## block_meditation_analyzer.py
"""
This script is the core engine of the neurofeedback experiment.

It manages the entire session, from data acquisition and signal processing to
real-time analysis and triggering feedback. It operates without a GUI, printing
all output directly to the console.
"""

import time
import numpy as np
import json
import os
import pylsl
from pylsl import StreamInlet, resolve_byprop
from scipy import signal
import threading
import subprocess
import sys
from block_metrics_analyzer import BlockMetricsAnalyzer 
import random
import pygame # Used for playing the main structural audio files (intro, conclusion, etc.).
import traceback
import socket # Used for sending UDP commands to the action_handler
import requests # Used for sending HTTP commands to the Epi


# --- Constants & Default Settings ---
# Default frequency bands for analysis.
THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)

# Default session structure.
DEFAULT_BLOCK_DURATION = 60
DEFAULT_BASELINE_BLOCKS = 3
DEFAULT_MEDITATION_BLOCKS = 15

# Muse 2 technical specifications.
SAMPLE_RATE = 256
CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]
# Focus is on posterior channels (TP9, TP10) where alpha is most prominent.
POSTERIOR_CHANNELS = [0, 3]

# Global debug flag.
DEBUG = True

def debug_print(*args, **kwargs):
    """A helper function to print messages only when DEBUG is True."""
    if DEBUG: print(*args, **kwargs)

# --- Main Analyzer Class ---

class BlockBasedMeditationAnalyzer:
    """
    Encapsulates all logic for running a neurofeedback meditation session.
    """
    def __init__(self,
                 baseline_blocks=DEFAULT_BASELINE_BLOCKS,
                 meditation_blocks=DEFAULT_MEDITATION_BLOCKS,
                 block_duration=DEFAULT_BLOCK_DURATION,
                 voice_type="human",
                 audio_path="audio_manuscript",
                 debug_mode=DEBUG,
                 ipc_host="127.0.0.1",
                 ipc_port=12345):
        """Initializes the analyzer with session parameters and sets up components."""

        global DEBUG
        DEBUG = debug_mode
        debug_print("Initializing BlockBasedMeditationAnalyzer...")

        # --- Session Structure ---
        self.sample_rate = SAMPLE_RATE
        self.baseline_duration_blocks = baseline_blocks
        self.meditation_duration_blocks = meditation_blocks
        self.block_duration_sec = block_duration
        self.total_session_blocks = self.baseline_duration_blocks + self.meditation_duration_blocks
        self.block_buffer_size = int(self.block_duration_sec * self.sample_rate)

        # --- Data buffers & State Tracking ---
        self.block_buffer = np.zeros((len(CHANNEL_NAMES), self.block_buffer_size))
        self.block_timestamps = np.zeros(self.block_buffer_size)
        self.block_samples_collected = 0
        self.current_block = 0
        self.current_phase = "setup"
        self.block_results = [] # Stores analysis results from every block.

        # --- Baseline & Threshold Variables ---
        # These are used to personalize feedback thresholds after the baseline phase.
        self.baseline_alpha_avg = 0
        self.baseline_theta_avg = 0
        self.baseline_ratio_avg = 0
        self.baseline_ratio_var_avg = 0
        self.baseline_complete = False
        self.baseline_blocks_data = [] 
        self.alpha_theta_ratio_threshold = 0
        self.ratio_variance_threshold = 0

        # --- Signal Processing Settings ---
        self.use_filters = True 
        self.bandpass_low = 3.0
        self.bandpass_high = 30.0 
        self.notch_freq = 50.0
        self.metrics_analyzer = BlockMetricsAnalyzer(sample_rate=self.sample_rate)
        self.stop_flag = False
        self.paused = False

        # --- Audio & IPC Configuration ---
        self.voice_type = voice_type
        self.audio_path = audio_path
        self.intro_audio = "Introduction.mp3"
        self.baseline_instruction_audio = "Baseline_instruction.mp3"
        self.podcast_audio = "3_minute_podcast.mp3"
        self.meditation_instruction_audio = "Meditation_instruction.mp3"
        self.conclusion_audio = "Conclusion.mp3"
        self.initial_prompt_filename = "initial_statement_1.mp3"
        self.num_variations = {
            "High Variance": 6, "Meditative": 6, "Relaxed": 6, "Sleepy": 6
        }
        
        # Setup UDP socket to send 'PLAY' commands to the action_handler.
        self.ipc_host = ipc_host
        self.ipc_port = ipc_port
        try:
            self.ipc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            debug_print(f"UDP Command Socket created for {self.ipc_host}:{self.ipc_port}")
        except Exception as e:
            print(f"FATAL: Could not create UDP socket: {e}")
            raise

        # --- Feedback Timing Control ---
        # Rules to prevent feedback from being delivered too frequently.
        self.last_feedback_block = -1
        self.last_meditative_feedback_block = -1
        self.minimum_interval_blocks = 3    # General cooldown for any feedback.
        self.meditative_interval_blocks = 4 # Specific, longer cooldown for 'Meditative' feedback.
        self.previous_detected_state = None

        # Initialize Pygame Mixer for playing main instructional audio.
        try:
            pygame.mixer.init()
            debug_print("Pygame mixer initialized successfully.")
            self.mixer_initialized = True
        except Exception as e:
            print(f"Error initializing pygame mixer: {e}")
            print("Structural audio playback may fail.")
            self.mixer_initialized = False

        self.start_time = None
        self.design_filters()


    def _send_epi_command_thread(self, command_url):
        """Sends an HTTP GET request to Epi in a non-blocking thread."""
        try:
            debug_print(f"Sending command to Epi (Thread): {command_url}")
            response = requests.get(command_url, timeout=2.0)
            if response.status_code == 200:
                debug_print(f"Epi command success (Status: {response.status_code})")
            else:
                print(f"Warning: Epi command failed. Status: {response.status_code}, Response: {response.text}")
        except requests.exceptions.RequestException as e:
             print(f"Warning: An error occurred sending Epi command: {e}")

    def send_epi_command(self, command_url):
        """Dispatches Epi command to a new thread to avoid blocking."""
        debug_print(f"Dispatching Epi command thread for: {command_url}")
        thread = threading.Thread(target=self._send_epi_command_thread, args=(command_url,), daemon=True)
        thread.start()

    def play_audio(self, filepath):
        """Loads and plays a main instructional audio file using pygame."""
        if not self.mixer_initialized:
            print("Audio Error: Mixer not initialized.")
            return False
        # Stop any currently playing instruction before starting the new one.
        if pygame.mixer.music.get_busy():
            debug_print(f"Audio Warning: Mixer busy, stopping previous sound to play {os.path.basename(filepath)}")
            pygame.mixer.music.stop()
            time.sleep(0.1)
        if not os.path.exists(filepath):
            print(f"Audio Error: File not found - {filepath}")
            return False
        try:
            pygame.mixer.music.load(filepath)
            debug_print(f"Playing structural audio: {filepath}")
            pygame.mixer.music.play()
            return True
        except pygame.error as e:
            print(f"Audio Error: Failed to load or play {filepath} - {e}")
            return False

    def send_feedback_command(self, state_name, variation_number):
        """Sends a UDP command to the action_handler to play a feedback prompt."""
        command = f"PLAY:{state_name}:{variation_number}"
        try:
            debug_print(f"Sending command: {command} to {self.ipc_host}:{self.ipc_port}")
            self.ipc_socket.sendto(command.encode('utf-8'), (self.ipc_host, self.ipc_port))
        except socket.error as e:
            print(f"Error sending UDP command '{command}': {e}")

    def design_filters(self):
        """Designs the bandpass and notch filters for efficient reuse."""
        nyquist = 0.5 * self.sample_rate
        # Bandpass filter to isolate frequencies of interest (3-30 Hz).
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist
        self.bandpass_b, self.bandpass_a = signal.butter(4, [low, high], btype='band')
        # Notch filter to remove 50 Hz electrical noise.
        notch_freq_norm = self.notch_freq / nyquist
        self.notch_b, self.notch_a = signal.iirnotch(notch_freq_norm, 30.0)
        debug_print("Filters designed.")

    def apply_filters(self, data):
        """Applies pre-designed filters to a chunk of EEG data."""
        filtered_data = data.copy()
        if self.use_filters:
            # Check for valid data shape to prevent filtering errors.
            if data.ndim != 2 or data.shape[1] < 151: # padlen=150 requires at least 151 points
                 debug_print(f"Skipping filtering: Invalid data shape {data.shape}")
                 return data
            # Use filtfilt for zero-phase filtering (avoids distorting the signal).
            for i in range(data.shape[0]):
                try:
                    filtered_data[i] = signal.filtfilt(self.bandpass_b, self.bandpass_a, filtered_data[i], padlen=150)
                    filtered_data[i] = signal.filtfilt(self.notch_b, self.notch_a, filtered_data[i], padlen=150)
                except ValueError as e:
                    print(f"Filtering error on channel {i}: {e}. Returning unfiltered.")
                    return data[i]
        return filtered_data
    
    def connect_to_muse(self):
        """Finds and connects to the Muse EEG stream on the network via LSL."""
        print("Resolving EEG stream...")
        streams = resolve_byprop('type', 'EEG', timeout=5.0)
        if not streams:
            raise RuntimeError("LSL Error: Can't find an EEG stream. Is MuseLSL running?")
        print("✓ EEG Stream found. Connecting...")
        self.inlet = StreamInlet(streams[0], max_chunklen=12)
        info = self.inlet.info()
        print(f"✓ Connected to: {info.name()} @ {info.nominal_srate()}Hz")

    def update_buffers(self, chunk, timestamps):
        """Adds a new chunk of EEG data to the current block's buffer."""
        # Only buffer data during the 'baseline' and 'meditation' phases.
        if self.current_phase not in ["baseline", "meditation"]:
            return

        chunk_size = chunk.shape[0]
        # Transpose chunk to (channels x samples) and select all channels for buffering.
        chunk_data = chunk[:, :len(CHANNEL_NAMES)].T
        space_left = self.block_buffer_size - self.block_samples_collected
        samples_to_add = min(chunk_size, space_left)

        if samples_to_add > 0:
            start_idx = self.block_samples_collected
            end_idx = start_idx + samples_to_add
            # Add the new data to the block buffer.
            self.block_buffer[:, start_idx : end_idx] = chunk_data[:, :samples_to_add]
            self.block_samples_collected += samples_to_add

    def calculate_baseline_stats(self):
        """Calculates average metrics from the baseline to set personalized thresholds."""
        if not self.baseline_blocks_data:
            print("\nWarning: No baseline data to calculate stats from.")
            self.baseline_complete = False
            return
        
        # Filter out any blocks where metrics might have failed.
        valid_baseline_blocks = [b for b in self.baseline_blocks_data if b.get('ratio_variance', -1) >= 0]
        if not valid_baseline_blocks:
            print("\nWarning: No valid baseline blocks to calculate stats.")
            self.baseline_complete = False
            return

        # Calculate the average ratio and variance from all valid baseline blocks.
        self.baseline_ratio_avg = np.mean([b['alpha_theta_ratio'] for b in valid_baseline_blocks])
        self.baseline_ratio_var_avg = np.mean([b['ratio_variance'] for b in valid_baseline_blocks])
        
        # Set the personalized thresholds for the meditation phase.
        # Meditative state threshold = 115% of baseline average ratio.
        self.alpha_theta_ratio_threshold = self.baseline_ratio_avg * 1.15
        # Stability threshold = 85% of baseline average variance (lower is more stable).
        epsilon = 1e-9 # Prevent threshold from being zero.
        self.ratio_variance_threshold = max(epsilon, self.baseline_ratio_var_avg * 0.85)

        print("\n--- Baseline Statistics Calculated ---")
        print(f"  Avg α/θ Ratio: {self.baseline_ratio_avg:.4f} | Avg Ratio Variance: {self.baseline_ratio_var_avg:.6f}")
        print(f"  Thresholds Set: Ratio > {self.alpha_theta_ratio_threshold:.4f}, Variance < {self.ratio_variance_threshold:.6f}")
        print("------------------------------------")
        self.baseline_complete = True

    def transition_to_meditation_phase(self):
        """Handles the state transition from instructions to the meditation phase."""
        self.current_phase = "meditation"
        print("\n===== TRANSITIONING TO MEDITATION PHASE =====")
        if not self.baseline_complete:
            print("Warning: Baseline incomplete, using default thresholds which may be inaccurate.")
        print(f"Meditation phase will consist of {self.meditation_duration_blocks} blocks.")

    def analyze_completed_block(self):
        """Analyzes a 60s block, detects the user's state, and triggers feedback."""
        block_num_display = self.current_block + 1
        debug_print(f"\n----- Analyzing Block {block_num_display} ({self.current_phase}) -----")

        # Ensure there is enough data for a meaningful analysis.
        if self.block_samples_collected < self.sample_rate:
             print(f"Warning: Block {block_num_display} has insufficient data. Skipping analysis.")
             return
        
        # Process the buffered data for this block.
        full_block_data = self.block_buffer[:, :self.block_samples_collected]
        filtered_block = self.apply_filters(full_block_data)
        
        # Average the two posterior channels (TP9, TP10) for the final analysis.
        posterior_data = filtered_block[POSTERIOR_CHANNELS, :]
        avg_posterior = np.mean(posterior_data, axis=0)
        
        metrics = self.metrics_analyzer.analyze_block(avg_posterior)
        metrics.update({'block_number': block_num_display, 'phase': self.current_phase})

        # Store and display results.
        self.block_results.append(metrics)
        if self.current_phase == "baseline":
             self.baseline_blocks_data.append(metrics)
        print(f"  Block {block_num_display} Metrics: α/θ Ratio: {metrics.get('alpha_theta_ratio', 0):.2f}, Var: {metrics.get('ratio_variance', -1):.4f}")

        # --- Adaptive Feedback Logic (only runs during meditation) ---
        if self.current_phase == "meditation" and self.baseline_complete:
            # On the first meditation block, play an initial prompt to orient the user.
            if self.current_block == self.baseline_duration_blocks:
                self.play_audio(os.path.join(self.audio_path, self.initial_prompt_filename))
                self.last_feedback_block = block_num_display
                self.previous_detected_state = "Initial Prompt"
                return

            # --- State Detection ---
            # Compare the block's metrics to the personalized thresholds.
            detected_state = None
            ratio_variance = metrics.get('ratio_variance', -1)
            alpha_theta_ratio = metrics.get('alpha_theta_ratio', 0)
            
            if ratio_variance < 0:
                print(f"Warning: Invalid variance for Block {block_num_display}. Skipping state detection.")
            elif ratio_variance > self.ratio_variance_threshold:
                detected_state = "High Variance" # Unstable state.
            elif alpha_theta_ratio >= self.alpha_theta_ratio_threshold:
                detected_state = "Meditative" # Focused state.
            elif self.baseline_ratio_avg <= alpha_theta_ratio < self.alpha_theta_ratio_threshold:
                detected_state = "Relaxed" # Relaxed but not fully meditative.
            else:
                detected_state = "Sleepy" # Low ratio may indicate drowsiness.

            # --- Feedback Timing Rules ---
            should_give_feedback = False
            if detected_state:
                blocks_since_any = block_num_display - self.last_feedback_block
                # General cooldown: at least 3 blocks must pass.
                if blocks_since_any >= self.minimum_interval_blocks:
                    # 'Meditative' state has a longer, specific cooldown.
                    if detected_state == "Meditative":
                        blocks_since_meditative = block_num_display - self.last_meditative_feedback_block
                        if blocks_since_meditative >= self.meditative_interval_blocks:
                            should_give_feedback = True
                    else:
                        should_give_feedback = True # Other states can get feedback if general cooldown is met.

            # --- Send Command ---
            if should_give_feedback:
                variation = random.randint(1, self.num_variations.get(detected_state, 1))
                self.send_feedback_command(detected_state, variation)
                # Update timers after sending feedback.
                self.last_feedback_block = block_num_display
                if detected_state == "Meditative":
                    self.last_meditative_feedback_block = block_num_display
                debug_print(f"Feedback sent for '{detected_state}'. Timers updated.")
            self.previous_detected_state = detected_state

    def run(self):
        """The main entry point that runs the entire session state machine."""
        self.session_completed_normally = False

        # Durations for instructional audio phases.
        intro_duration = 78 + 1
        baseline_instr_duration = 30 + 1
        meditation_instr_duration = 113 + 1
        conclusion_duration = 53 + 1

        try:
            self.connect_to_muse()
            
            # --- Start Session: Introduction ---
            self.current_phase = "Introduction"
            self.phase_start_time = time.time()
            phase_end_time = self.phase_start_time + intro_duration
            print("\nPhase: Introduction")
            self.send_epi_command("http://127.0.0.1:8000/command/SR.trig/1/0/0")
            self.play_audio(os.path.join(self.audio_path, self.intro_audio))
            
            # --- Main Application Loop ---
            while not self.stop_flag:
                current_time = time.time()
                
                # Pull a small chunk of data from the LSL stream.
                chunk, timestamps = self.inlet.pull_chunk(timeout=0.1, max_samples=int(self.sample_rate * 0.2))
                if timestamps and len(chunk) > 0:
                    self.update_buffers(np.array(chunk), timestamps)

                # --- State Machine Logic ---
                phase_has_ended = current_time >= phase_end_time if phase_end_time else False

                # == Intro -> Baseline Instructions ==
                if self.current_phase == "Introduction" and phase_has_ended:
                    self.current_phase = "Baseline Instructions"
                    phase_end_time = current_time + baseline_instr_duration
                    print("\nPhase: Baseline Instructions")
                    self.send_epi_command("http://127.0.0.1:8000/command/SR.trig/2/0/0")
                    self.play_audio(os.path.join(self.audio_path, self.baseline_instruction_audio))

                # == Baseline Instructions -> Baseline Recording ==
                elif self.current_phase == "Baseline Instructions" and phase_has_ended:
                    self.current_phase = "baseline"
                    self.current_block = 0
                    self.block_start_time = current_time
                    phase_end_time = None # This phase is driven by block count.
                    print("\nPhase: Baseline Recording")
                    self.play_audio(os.path.join(self.audio_path, self.podcast_audio))

                # == Baseline Recording Phase ==
                elif self.current_phase == "baseline":
                    if self.block_start_time and (current_time - self.block_start_time >= self.block_duration_sec):
                        self.analyze_completed_block()
                        # Reset for next block
                        self.block_buffer.fill(0)
                        self.block_timestamps.fill(0)
                        self.block_samples_collected = 0
                        self.current_block += 1
                        self.block_start_time = current_time

                        # Check if baseline phase is over
                        if self.current_block == self.baseline_duration_blocks:
                            if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()
                            self.calculate_baseline_stats()
                            
                            self.current_phase = "Meditation Instructions"
                            phase_end_time = current_time + meditation_instr_duration
                            print("\nPhase: Meditation Instructions")
                            self.send_epi_command("http://127.0.0.1:8000/command/SR.trig/3/0/0")
                            self.play_audio(os.path.join(self.audio_path, self.meditation_instruction_audio))
                        else:
                            print(f"Starting Baseline Block {self.current_block + 1}...")

                # == Meditation Instructions -> Meditation ==
                elif self.current_phase == "Meditation Instructions" and phase_has_ended:
                    self.transition_to_meditation_phase()
                    self.block_start_time = current_time
                    phase_end_time = None

                # == Meditation Phase ==
                elif self.current_phase == "meditation":
                    if self.block_start_time and (current_time - self.block_start_time >= self.block_duration_sec):
                        self.analyze_completed_block()
                        self.block_buffer.fill(0)
                        self.block_timestamps.fill(0)
                        self.block_samples_collected = 0
                        self.current_block += 1
                        self.block_start_time = current_time

                        # Check if meditation phase is over
                        if self.current_block == self.total_session_blocks:
                            self.current_phase = "Conclusion"
                            phase_end_time = current_time + conclusion_duration
                            print("\nPhase: Conclusion")
                            self.send_epi_command("http://127.0.0.1:8000/command/SR.trig/4/0/0")
                            self.play_audio(os.path.join(self.audio_path, self.conclusion_audio))
                        else:
                            med_block_num = self.current_block - self.baseline_duration_blocks + 1
                            print(f"Starting Meditation Block {med_block_num}...")

                # == Conclusion -> End Session ==
                elif self.current_phase == "Conclusion" and phase_has_ended:
                    print("Conclusion audio finished.")
                    self.stop_flag = True
                    self.session_completed_normally = True # Set flag to allow saving results.

                time.sleep(0.02) # Prevent busy-waiting.
            
            # --- Conditional Saving ---
            # Save results only if the session finished normally, not if it was aborted.
            if self.session_completed_normally:
                print("\nSession completed normally. Saving results...")
                self.save_results()
            else:
                print("\nSession was interrupted. Results will NOT be saved.")

        except Exception as e:
            print(f"\nAn unexpected error occurred during the session: {e}")
            traceback.print_exc()
        finally:
            # --- Final Cleanup ---
            print("\nStarting final cleanup...")
            if self.mixer_initialized and pygame.mixer.get_init():
                pygame.mixer.quit()
            if hasattr(self, 'inlet') and self.inlet:
                self.inlet.close_stream()
            if hasattr(self, 'ipc_socket'):
                self.ipc_socket.close()
            print("\nCleanup finished.")
            sys.exit(0)

    def save_results(self):
        """Saves all session data and parameters to a timestamped JSON file."""
        if not self.block_results:
            print("Warning: No results to save.")
            return

        # Define the target directory for saving results.
        data_dir = "/Users/epi/Documents/epi_in_serenity/Claud/meditation_results"
        os.makedirs(data_dir, exist_ok=True)
        
        # Compile all relevant session data.
        results = {
            'session_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'config': {
                'baseline_blocks': self.baseline_duration_blocks,
                'meditation_blocks': self.meditation_duration_blocks
            },
            'baseline_summary': {
                'avg_ratio': self.baseline_ratio_avg,
                'avg_variance': self.baseline_ratio_var_avg
            },
            'thresholds_set': {
                'ratio_threshold': self.alpha_theta_ratio_threshold,
                'variance_threshold': self.ratio_variance_threshold
            },
            'block_data': self.block_results
        }
        
        filename = os.path.join(data_dir, f"meditation_session_{time.strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=4, cls=NumpyEncoder)
            print(f"✓ Results successfully saved to {filename}")
        except Exception as e:
            print(f"FATAL: Error saving results to JSON file: {e}")

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

if __name__ == "__main__":
    # This block is for direct testing. The intended use is via 'main.py'.
    print("Warning: Running directly. Ensure 'action_handler.py' is running separately.")
    # Setup argparse for direct execution
    parser = argparse.ArgumentParser(description="Block-Based Meditation Analyzer (No GUI)")
    parser.add_argument('--debug', action='store_true', default=DEBUG)
    parser.add_argument('--blocks', type=int, default=DEFAULT_BASELINE_BLOCKS)
    parser.add_argument('--meditation', type=int, default=DEFAULT_MEDITATION_BLOCKS)
    args = parser.parse_args()

    analyzer = BlockBasedMeditationAnalyzer(
        baseline_blocks=args.blocks,
        meditation_blocks=args.meditation,
        debug_mode=args.debug
    )
    analyzer.run()