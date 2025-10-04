# VIP-PV-SOM Microcircuit Neural Network Simulation

## Overview
This project implements a cortical microcircuit model based on Izhikevich neural networks to investigate how VIP, PV (Parvalbumin), and SOM (Somatostatin) interneurons regulate noise correlations through feedforward and feedback inhibition mechanisms.

## Features
- Complete cortical microcircuit with 3-layer architecture
- VIP disinhibition circuit implementation
- Feedforward (FF-Inh) and Feedback (FB-Inh) inhibition analysis
- Noise correlation and synchrony analysis
- Common input strength calculation for variable-length time series

## Requirements
- MATLAB R2018b or later
- Signal Processing Toolbox (for Hilbert transform in synchrony analysis)
- Statistics and Machine Learning Toolbox (for PCA analysis)

## File Structure
├── vip_pv_som_feedback_inh.m # Feedback inhibition experiment
├── vip_pv_som_feedforward_inh.m # Feedforward inhibition experiment
├── vip_pv_som_disinhibition.m # VIP disinhibition experiment
├── test_inhibitory_population_systematic_E.m # Systematic testing of the effects of inhibitory neuron count on network performance
├── LICENSE
└── docs/
├── README.md
## Quick Start
```matlab
% Run feedback inhibition experiment
vip_pv_som_feedback_inh

% Run feedforward inhibition experiment
vip_pv_som_feedforward_inh

% Run VIP disinhibition experiment
vip_pv_som_disinhibition

% Run Systematic testing of the effects of inhibitory neuron count on network performance
test_inhibitory_population_systematic_E

## Network Architecture
Layer 1 (L1): 8 excitatory + 2 inhibitory neurons
Layer 2 (L2): 12 excitatory + 3 inhibitory neurons
Layer 3 (L3): 20 excitatory + 4 FF-Inh + 4 FB-Inh neurons
VIP neurons: 6 neurons

## The scripts generate:
comprehensive analysis figures with error bars
synchrony analysis plots
common input strength analysis
save .mat files with detailed results
PNG and FIG format figures

Author
Guangwei Xu
