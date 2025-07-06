# Multispeaker LCMV Beamformer with Postfilter for Source Separation and Noise Reduction

This repository contains an implementation of a **Multispeaker Linearly Constrained Minimum Variance (LCMV) Beamformer** combined with a **multispeaker Wiener or LSA postfilter**, aimed at source separation and noise reduction in multi-microphone audio recordings.

---

## Objective

To enhance speech signals from multiple speakers in a noisy environment by:
- Separating speaker signals using spatial diversity
- Reducing residual noise using statistical postfiltering

---

## Methodology

This implementation is based on the two-stage MMSE decomposition proposed in the IEEE/ACM paper:

**Stage 1: Multispeaker LCMV Beamformer**
- Utilizes spatial constraints and microphone array signals to isolate individual speaker sources.
- Implemented using either direct formulation or the Generalized Sidelobe Canceller (GSC) structure.

**Stage 2: Multispeaker Postfilter**
- Applies a Wiener or Log Spectral Amplitude (LSA) filter on the beamformer output.
- Enhances signal quality by suppressing residual noise and interference.

---

## Features

-  STFT-based multichannel signal processing
-  RTF (Relative Transfer Function) estimation via GEVD
-  LCMV Beamforming using constrained optimization
-  GSC-structured beamforming for efficient implementation
-  Multi-speaker Wiener and LSA postfilters
-  Decision-directed a priori SNR estimation
-  Output is a set of clean, separated signals (1 per speaker)



