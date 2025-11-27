
# NG GNSS Class: Fall 2025

This repository contains the final project for the NG GNSS class held in Fall 2025. The project focuses on developing code that acquires and tracks GPS L1 C/A signals.

## Project Overview

The main objective of this project is to implement a software receiver that can acquire and track GPS L1 C/A signals. The project is divided into several key components:

1. **Signal Acquisition**: Implement algorithms to detect the presence of GPS signals and determine their code phase and Doppler frequency.

2. **Signal Tracking**: Develop tracking loops (e.g., Phase-Locked Loop, Delay-Locked Loop) to maintain lock on the acquired signals and extract navigation data.

3. **Performance Evaluation**: Analyze the performance of the acquisition and tracking algorithms under various conditions, such as different signal-to-noise ratios and simulated signal dynamics.


## Datasets

We will use both simulated and real datasets for testing and validation.  The real dataset (60s) will be provided on-site.  The simulated datasets (recommended 2-5s) we will generate using code that we developed as part of Lecture 8 exercises.  In particular, we want to simulate signals with different receiver dynamics (stationary, constant velocity, acceleration, and jerk) at relatively low C/N0 in order to demonstrate tracking loop performance and the potential for loss-of-lock in mis-tuned carrier tracking loops.

## Project Structure

We will divide the project code into an acquisition stage and a tracking stage, saving our acquisition results in between in order to reduce computational burden.  I have set up the code examples to output results into a `local-data` folder in the project root directory.