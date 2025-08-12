#!/usr/bin/env python
# -*- coding: utf-8 -*-

## block_metrics_analyzer.py
"""
A specialized module for calculating neurofeedback metrics from EEG data.

This script defines the 'BlockMetricsAnalyzer' class, which is imported by the
main analyzer. Its sole responsibility is to take a block of raw EEG data
and compute the key metrics used for the experiment:
1. The overall Alpha/Theta power ratio.
2. The variance of the Alpha/Theta ratio over time (as a stability metric).
"""

import numpy as np
from scipy import signal

class BlockMetricsAnalyzer:
    """
    Encapsulates the algorithms for EEG signal analysis.
    """
    def __init__(self, sample_rate=256):
        """
        Initializes the analyzer with parameters for signal processing.

        Args:
            sample_rate (int): The EEG sample rate in Hz.
        """
        self.sample_rate = sample_rate

        # --- Analysis Parameters ---
        self.theta_band = (4.0, 8.0)
        self.alpha_band = (8.0, 13.0)

        # Welch's method is used for Power Spectral Density (PSD) estimation.
        # 'nperseg' defines the length of each segment used in the calculation.
        self.nperseg = int(2 * self.sample_rate) # 2-second segments.

        # For the stability metric, the 60-second block is broken into smaller sub-windows.
        sub_window_duration_sec = 5
        self.sub_window_size = int(sub_window_duration_sec * self.sample_rate)
        self.sub_window_overlap = int(self.sub_window_size * 0.5) # 50% overlap.


    def calculate_band_power(self, data, band):
        """
        Calculates the average power within a specific frequency band.

        Args:
            data (np.ndarray): 1D array of EEG data.
            band (tuple): A tuple defining the low and high frequency of the band.

        Returns:
            float: The average power in the specified band.
        """
        # Welch's method estimates power by averaging the PSD of overlapping segments.
        freqs, psd = signal.welch(data, fs=self.sample_rate, nperseg=self.nperseg)

        # Find the frequency indices that fall within the requested band.
        idx_band = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
        
        # Return the mean power within those frequencies.
        return np.mean(psd[idx_band])

    def calculate_alpha_theta_ratio(self, data):
        """
        Calculates the Alpha/Theta power ratio for a segment of data.

        Args:
            data (np.ndarray): 1D array of EEG data.

        Returns:
            tuple: (alpha_theta_ratio, alpha_power, theta_power)
        """
        alpha_power = self.calculate_band_power(data, self.alpha_band)
        theta_power = self.calculate_band_power(data, self.theta_band)

        # To avoid division by zero, we check if theta power is negligible.
        if theta_power < 1e-10:
            return 0.0, alpha_power, theta_power
        
        return alpha_power / theta_power, alpha_power, theta_power

    def analyze_block(self, block_data):
        """
        Analyzes a full block of EEG data to compute the primary and secondary metrics.

        Args:
            block_data (np.ndarray): 1D array of EEG data for a complete block (e.g., 60s).

        Returns:
            dict: A dictionary containing the calculated metrics for the block.
        """
        if block_data.size < self.nperseg:
             print("Warning: Not enough data in block to perform analysis.")
             # Return a dictionary with an error value for variance.
             return {'ratio_variance': -1}

        # --- Metric 1: Overall Block Ratio ---
        # First, calculate the alpha/theta ratio for the entire block.
        overall_ratio, overall_alpha, overall_theta = self.calculate_alpha_theta_ratio(block_data)

        # --- Metric 2: Ratio Variance (Stability) ---
        # To measure stability, calculate the ratio for smaller, overlapping sub-windows.
        sub_window_ratios = []
        start_index = 0
        while start_index + self.sub_window_size <= block_data.size:
            sub_data = block_data[start_index : start_index + self.sub_window_size]
            sub_ratio, _, _ = self.calculate_alpha_theta_ratio(sub_data)
            sub_window_ratios.append(sub_ratio)
            # Move the window forward by half its size for 50% overlap.
            start_index += self.sub_window_overlap

        # The variance of these sub-window ratios is our stability metric.
        # Lower variance indicates a more stable state.
        ratio_variance = np.var(sub_window_ratios) if len(sub_window_ratios) > 1 else -1

        # Return all computed metrics in a dictionary.
        return {
            'alpha_power': overall_alpha,
            'theta_power': overall_theta,
            'alpha_theta_ratio': overall_ratio,
            'ratio_variance': ratio_variance, # -1 indicates an error or insufficient data.
            'sub_window_ratios': sub_window_ratios
        }


# This block is for example usage and testing of the class.
if __name__ == "__main__":
    # Create a dummy 60-second EEG signal for testing.
    sample_rate = 256
    duration = 60
    num_samples = sample_rate * duration
    # Generate a signal with mixed frequencies.
    time = np.arange(num_samples) / sample_rate
    # A mix of a 10Hz alpha wave and a 6Hz theta wave.
    eeg_signal = np.sin(2 * np.pi * 10 * time) + 0.5 * np.sin(2 * np.pi * 6 * time) + 0.5 * np.random.randn(num_samples)
    
    # Initialize the analyzer and run the analysis.
    analyzer = BlockMetricsAnalyzer(sample_rate=sample_rate)
    metrics = analyzer.analyze_block(eeg_signal)

    print("--- Block Metrics Analyzer Example ---")
    print(f"Overall Alpha/Theta Ratio: {metrics['alpha_theta_ratio']:.4f}")
    print(f"Ratio Variance (Stability): {metrics['ratio_variance']:.4f}")
    print(f"Number of sub-windows analyzed: {len(metrics['sub_window_ratios'])}")
    print("--------------------------------------")