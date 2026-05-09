# GEMLA Architecture Card

## Name

GEMLA: Γ–EML–α Transport Architecture SDK

## Current Release Scope

GEMLA is a research SDK for finite, observable transport diagnostics. It evaluates whether a trajectory produces a stable lifted-phase transport signature that passes gates and rejects adversarial controls.

## Core Pipeline

source trajectory
→ surrogate trajectory
→ complex lifted observable
→ lifted phase Θ(t)
→ RevA v2-SF gate
→ adversarial controls
→ anchor diagnostics
→ winding diagnostics
→ PASS / FAIL verdict

Core Components
Component	Role
Γ operator	            Minimum-action / MAP update
EML class	            Symbolic representation substrate
α action	            Local path/action weight
Lifted phase	        Continuous phase observable
RevA v2-SF	            Orientation and spectral-flatness gate
Controls	            Wrong-sign, shuffle, scramble rejection
Anchors	                Event-irregularity diagnostic
Winding	                Integer transport-cell diagnostic

# Supported Inputs

Current release supports:

- synthetic transport trajectories
- synthetic industrial telemetry
- synthetic market microstructure
- synthetic cyber event transport
- latent embedding trajectories
- V-JEPA-style external embedding arrays saved as .npy
- Main Outputs
- final transport verdict
- RevA v2-SF pass/fail
- spectral-flatness score
- control rejection status
- anchor count and spacing CV
- winding jumps and winding cells
- Markdown reports
- Intended Use

GEMLA is intended for research, prototyping, and architecture evaluation of multiscale transport diagnostics.

# Not Intended For

GEMLA is not intended for autonomous high-stakes decision-making without independent validation, domain calibration, and human review.

# Current Maturity

Research preview.

The SDK currently demonstrates executable architecture behavior, benchmark reproducibility, latent embedding compatibility, and practical synthetic deployment examples.