# Epi in Serenity: A Neurofeedback Meditation System

This repository contains the complete Python software system developed for the Master's thesis, *"Brainwaves, Breath and Robots: Rethinking Feedback in Meditation Training"* The software is designed to run a real-time neurofeedback experiment comparing different modalities of guidance for Focused Attention (FA) meditation.

The central inquiry of the thesis is to determine whether the physical presence of an embodied guide (a socially assistive robot) is more effective at helping a meditator regulate their brain state than a disembodied voice. 


---
## System Architecture

The system is composed of several interconnected Python scripts that work together to run an experimental session:

1.  **Muse LSL Streamer (`start_muse.sh`):** A shell script that uses `muselsl` to stream EEG data from a Muse 2 headband onto the local network via Lab Streaming Layer (LSL). 
2.  **Main Controller (`main.py`):** The entry point of the application. It parses settings, starts the `action_handler.py` in the background, and then launches the main analyzer. 
3.  **Session & Analysis Core (`block_meditation_analyzer.py`):** The heart of the system. It connects to the LSL stream, manages the session flow (intro, baseline, meditation), collects data in 60-second blocks, and orchestrates analysis and feedback. 
4.  **Metrics Calculator (`block_metrics_analyzer.py`):** A dedicated module for EEG signal processing. It takes blocks of data and calculates the two key metrics for the study:
    * **Alpha/Theta Ratio:** A marker for focused, internalized attention versus distraction or drowsiness.
    * **Ratio Variance:** A metric for attentional stability, calculated over 5-second sub-windows. Lower variance indicates higher stability. 
5.  **Audio Feedback Handler (`action_handler.py`):** A lightweight, background UDP server that receives commands from the main analyzer (e.g., `PLAY:Meditative:1`) and plays the corresponding audio file. This separation prevents audio playback from interfering with the main analysis loop.

---
## Features

* **Structured Session Flow:** Manages a complete meditation session with introduction, baseline recording (3-minute podcast clip is included), guided meditation, and conclusion phases. 
* **Real-time EEG Analysis:** Processes EEG data in 60-second blocks to provide timely feedback. 
* **Personalized Neurofeedback:** Establishes personalized thresholds for the alpha/theta ratio and stability based on a 3-minute baseline recording. 
* **Adaptive Feedback:** Categorizes the user's brain state into 'High Variance', 'Meditative', 'Relaxed', or 'Sleepy' and provides state-specific audio prompts.
* **Multi-modal Commands:** Sends UDP commands for audio feedback and HTTP commands to control a robot named Epi.
* **Configurable & Extensible:** Session parameters (durations, blocks, etc.) can be easily configured via command-line arguments or modifications in the corresponding script.

---
## Requirements

### Hardware
* Muse 2 EEG Headband 
* (Optional) Epi Humanoid Robot for the embodied feedback condition. 
* A computer capable of running Python and connecting to the Muse via Bluetooth.

### Software
* Python 3.7+
* The Python libraries listed in `requirements.txt`.

---
## Installation

1.  **Clone the repository:**
    Open your terminal, and run the following commands to download the project and navigate into its folder.
    ```bash
    git clone [https://github.com/danderstorm1/Epi_in_Serenity.git](https://github.com/danderstorm1/Epi_in_Serenity.git)
    cd Epi_in_Serenity
    ```

2.  **Create a virtual environment (recommended):**
    This isolates the project's dependencies from your system's main Python installation.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    Create a new file named `requirements.txt` in your project folder, paste the contents below into it, and then run the install command.
    ```bash
    pip install -r requirements.txt
    ```

    **Contents of `requirements.txt`:**
    ```
    numpy
    scipy
    pylsl
    pygame
    requests
    muselsl
    simpleaudio
    sounddevice
    soundfile
    ```

---
## Usage

Running an experiment involves three steps:

1.  **Start the Muse LSL Stream:**
    Open a terminal, navigate to the project folder, and run the streaming script, to ensure the Muse 2 is connected.
    
    ```bash
    ./start_muse.sh
    ```
    A visualization window from `muselsl` should appear, confirming the stream is active. **Leave this terminal running in the background.** (If the software happens to crash, press Control + C in this terminal to stop it, then run ./start_muse.sh again to restart it. Be sure to leave the other terminals open).

3.  **Start the Epi or audio system:**

Open a **second** terminal window, navigate to the project directory, and run the following script, to activate the audio or Epi commands:

```bash

action_handler.py

```

3.  **Start the Software/Guide (Epi or disembodied guide):**
    Open a **third** terminal window and run the main script:

```bash

python main.py

```
    # Example: Run a shorter customized session 
    python main.py --meditation 5 --duration 30 
    ```
    * `--blocks`: Sets the number of baseline blocks.
    * `--meditation`: Sets the number of meditation blocks.
    * `--duration`: Sets the length of each analysis block in seconds.

---
## Code Overview

* `main.py`: The main entry point. It parses arguments and launches the other components.
* `block_meditation_analyzer.py`: The core controller that manages the entire experimental session from start to finish.
* `block_metrics_analyzer.py`: The signal processing engine that calculates the alpha/theta ratio and its variance.
* `action_handler.py`: The background audio player that listens for UDP commands.
* `start_muse.sh`: A helper script to begin streaming data from the Muse 2 headband.

---
## License

Do not have one yet.
