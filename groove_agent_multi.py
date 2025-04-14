import argparse
import os
import random
import time
from mido import Message, MidiFile, MidiTrack, MetaMessage
import numpy as np
from midi_to_wav import render_midi_with_sample_groups, group_samples_by_type

# --- Style Presets ---
# Define characteristics for different groove styles
# - note_probability: (min, max) chance of a note being placed at a step (scaled by density).
# - velocity_range: (min, max) MIDI velocity for the note.
# - timing_randomness_ms: (min, max) milliseconds deviation from the grid (scaled by density).
# - double_hit_probability: Chance of adding a quick second hit after the main one.
# - skip_hit_probability: Chance of removing a note that was initially placed.
# - ghost_note_probability: Chance of adding a low-velocity note in an empty step.
# - ghost_note_velocity_factor: Multiplier for ghost note velocity (e.g., 0.4 = 40% of normal).
# - swing_amount: 0.0 (straight) to ~0.66 (heavy swing), affects off-beat 16ths.
# - use_strict_pattern: If True, ignores probabilities/variations and uses hardcoded pattern logic.
# NEW KEY:
# - pattern_weights: List[16] floats (0.0-1.0) representing relative probability weight for each 16th note step in a bar.
STYLE_PRESETS = {
    "glitch": {
        "description": "Stuttering, broken rhythms with unexpected timing.",
        "kick": {
            "note_probability": (0.2, 0.5), 
            "velocity_range": (90, 110),
            "timing_randomness_ms": (5, 30), 
            "double_hit_probability": 0.20,
            "skip_hit_probability": 0.15,
            "ghost_note_probability": 0.05,
            "ghost_note_velocity_factor": 0.3,
            "pattern_weights": [0.6]*16,
        },
        "snare": {
            "note_probability": (0.05, 0.25),
            "velocity_range": (70, 90),
            "timing_randomness_ms": (10, 40),
            "double_hit_probability": 0.25,
            "skip_hit_probability": 0.10,
            "ghost_note_probability": 0.15,
            "ghost_note_velocity_factor": 0.4,
            "pattern_weights": [0.5]*16,
        },
        "closed_hat": {
            "note_probability": (0.4, 0.7),
            "velocity_range": (45, 70),
            "timing_randomness_ms": (3, 20),
            "double_hit_probability": 0.15,
            "skip_hit_probability": 0.30,
            "ghost_note_probability": 0.00,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.7, 0.8, 0.9, 0.7, 0.8, 0.6, 0.9, 0.8, 0.7, 0.9, 0.8, 0.6, 0.9, 0.7, 0.8, 0.9],
            "reverb_params": {"time_ms": 50, "mix": 0.05},
        },
        "open_hat": {
            "note_probability": (0.02, 0.15),
            "velocity_range": (55, 80),
            "timing_randomness_ms": (3, 20),
            "double_hit_probability": 0.00,
            "skip_hit_probability": 0.00,
            "ghost_note_probability": 0.00,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.3, 0.6, 0.4, 0.7, 0.2, 0.5, 0.3, 0.6, 0.4, 0.7, 0.2, 0.5, 0.3, 0.6, 0.4, 0.8],
            "reverb_params": {"time_ms": 120, "mix": 0.12},
        },
        "rim": {
            "note_probability": (0.05, 0.2),
            "velocity_range": (60, 80),
            "timing_randomness_ms": (10, 30),
            "double_hit_probability": 0.1,
            "skip_hit_probability": 0.2,
            "ghost_note_probability": 0.1,
            "ghost_note_velocity_factor": 0.4,
            "pattern_weights": [0.3]*16,
        },
        "clap": {
            "note_probability": (0.0, 0.05),
            "velocity_range": (80, 100),
            "timing_randomness_ms": (5, 20),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.1]*16,
        },
        "cowbell": {
            "note_probability": (0.0, 0.02),
            "velocity_range": (90, 100),
            "timing_randomness_ms": (0, 5),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.1]*16,
        },
        "claves": {
            "note_probability": (0.0, 0.03),
            "velocity_range": (85, 95),
            "timing_randomness_ms": (0, 2),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.05,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.1]*16,
        },
        "low_tom": {
            "note_probability": (0.01, 0.1),
            "velocity_range": (90, 110),
            "timing_randomness_ms": (5, 20),
            "double_hit_probability": 0.05,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.05,
            "ghost_note_velocity_factor": 0.6,
            "pattern_weights": [0.2]*16,
        },
        "mid_tom": {
            "note_probability": (0.01, 0.1),
            "velocity_range": (90, 110),
            "timing_randomness_ms": (5, 20),
            "double_hit_probability": 0.05,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.05,
            "ghost_note_velocity_factor": 0.6,
            "pattern_weights": [0.2]*16,
        },
        "high_tom": {
            "note_probability": (0.01, 0.1),
            "velocity_range": (90, 110),
            "timing_randomness_ms": (5, 20),
            "double_hit_probability": 0.05,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.05,
            "ghost_note_velocity_factor": 0.6,
            "pattern_weights": [0.2]*16,
        },
        "swing_amount": 0.0,
    },
    "dusty": {
        "description": "Lo-fi hip-hop style groove with swing and velocity variation.",
        "kick": {
            "note_probability": (0.25, 0.6),
            "velocity_range": (100, 127),
            "timing_randomness_ms": (0, 5),
            "double_hit_probability": 0.05,
            "skip_hit_probability": 0.05,
            "ghost_note_probability": 0.00,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.9,0.2,0.4,0.2, 0.6,0.1,0.5,0.1, 0.8,0.2,0.4,0.2, 0.6,0.1,0.5,0.1],
            "reverb_params": {"time_ms": 50, "mix": 0.05},
        },
        "snare": {
            "note_probability": (0.15, 0.3),
            "velocity_range": (65, 85),
            "timing_randomness_ms": (3, 15),
            "double_hit_probability": 0.10,
            "skip_hit_probability": 0.05,
            "ghost_note_probability": 0.20,
            "ghost_note_velocity_factor": 0.5,
            "pattern_weights": [0.1,0.1,0.1,0.1, 1.0,0.2,0.3,0.2],
            "reverb_params": {"time_ms": 150, "mix": 0.15},
        },
        "closed_hat": {
            "note_probability": (0.5, 0.8),
            "velocity_range": (35, 65),
            "timing_randomness_ms": (1, 10),
            "double_hit_probability": 0.08,
            "skip_hit_probability": 0.10,
            "ghost_note_probability": 0.00,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.8,0.7,0.8,0.6, 0.8,0.7,0.8,0.6, 0.8,0.7,0.8,0.6, 0.8,0.7,0.8,0.6],
            "reverb_params": {"time_ms": 80, "mix": 0.1},
        },
        "open_hat": {
            "note_probability": (0.05, 0.20),
            "velocity_range": (45, 70),
            "timing_randomness_ms": (1, 10),
            "double_hit_probability": 0.00,
            "skip_hit_probability": 0.00,
            "ghost_note_probability": 0.00,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.1,0.4,0.1,0.6, 0.1,0.4,0.1,0.7, 0.1,0.4,0.1,0.6, 0.1,0.4,0.1,0.8],
            "reverb_params": {"time_ms": 120, "mix": 0.12},
        },
        "rim": {
            "note_probability": (0.1, 0.3),
            "velocity_range": (50, 75),
            "timing_randomness_ms": (2, 10),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.2,
            "ghost_note_velocity_factor": 0.4,
            "pattern_weights": [0.1,0.1,0.5,0.1, 0.2,0.1,0.6,0.1, 0.1,0.1,0.5,0.1, 0.2,0.1,0.6,0.1],
            "reverb_params": {"time_ms": 100, "mix": 0.15},
        },
        "clap": {
            "note_probability": (0.05, 0.2),
            "velocity_range": (80, 100),
            "timing_randomness_ms": (3, 12),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.05,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.1,0.1,0.1,0.1, 0.7,0.1,0.1,0.1, 0.1,0.1,0.1,0.1, 0.7,0.1,0.1,0.1],
            "reverb_params": {"time_ms": 180, "mix": 0.2},
        },
        "cowbell": {
            "note_probability": (0.0, 0.01),
            "velocity_range": (90, 100),
            "timing_randomness_ms": (0, 5),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.0]*16,
        },
        "claves": {
            "note_probability": (0.0, 0.01),
            "velocity_range": (85, 95),
            "timing_randomness_ms": (0, 2),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.05,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.0]*16,
        },
        "low_tom": {
            "note_probability": (0.02, 0.1),
            "velocity_range": (85, 105),
            "timing_randomness_ms": (2, 10),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.3,0.1,0.1,0.1, 0.2,0.1,0.1,0.1, 0.3,0.1,0.1,0.1, 0.2,0.1,0.1,0.1],
            "reverb_params": {"time_ms": 200, "mix": 0.1},
        },
        "mid_tom": {
            "note_probability": (0.01, 0.08),
            "velocity_range": (85, 105),
            "timing_randomness_ms": (2, 10),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.1,0.1,0.1,0.2, 0.1,0.1,0.1,0.3, 0.1,0.1,0.1,0.2, 0.1,0.1,0.1,0.3],
            "reverb_params": {"time_ms": 180, "mix": 0.1},
        },
        "high_tom": {
            "note_probability": (0.01, 0.07),
            "velocity_range": (85, 105),
            "timing_randomness_ms": (2, 10),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.1,0.2,0.1,0.1, 0.1,0.3,0.1,0.1, 0.1,0.2,0.1,0.1, 0.1,0.3,0.1,0.1],
            "reverb_params": {"time_ms": 150, "mix": 0.1},
        },
        "swing_amount": 0.55, 
    },
    "house": { 
        "description": "Classic four-on-the-floor house beat with variations.",
        "use_strict_pattern": False,
        "kick":       {
            "note_probability": (0.9, 1.0), # Slightly reduced max prob
            "velocity_range": (110, 125), # Slightly softer max kick 
            "timing_randomness_ms": (0, 1), # Very tight kick
            "double_hit_probability": 0.0,     
            "skip_hit_probability": 0.01,    
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], # Strong 4/4
        },
        "snare":      { # Minimal snare, prefer clap
             "note_probability": (0.0, 0.05), # Very low chance
             "velocity_range": (80, 95),
             "timing_randomness_ms": (1, 4),
             "double_hit_probability": 0.0, 
             "skip_hit_probability": 0.6, # Likely skipped
             "ghost_note_probability": 0.0,
             "ghost_note_velocity_factor": 0.0,
             "pattern_weights": [0.1]*16, # Flat low weights
        },
         "clap":      { # Strong on 2 & 4
            "note_probability": (0.9, 1.0),
            "velocity_range": (95, 110), # Slightly softer max clap
            "timing_randomness_ms": (0, 2), # Tight clap
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.01,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], # Strongly on 2 & 4
             # Removed default reverb
        },
        "closed_hat": { # Strong off-beat 8ths
             "note_probability": (0.7, 0.9), # Reduced probability slightly
             "velocity_range": (45, 70), # Kept relatively soft
             "timing_randomness_ms": (0, 2), # Tight hats
             "double_hit_probability": 0.0, # No doubles usually
             "skip_hit_probability": 0.02, # Low skip
             "ghost_note_probability": 0.0,
             "ghost_note_velocity_factor": 0.0,
             "pattern_weights": [0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0], # Standard off-beats
              # Removed default delay
        },
        "open_hat":   { # Atmospheric punctuation
            "note_probability": (0.1, 0.3), # Lower probability
            "velocity_range": (60, 80),
            "timing_randomness_ms": (1, 5), # Tighter 
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.1, # Less likely to skip than before
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9], # End of 2-bar phrases mostly
            "reverb_params": {"time_ms": 250, "feedback": 0.1, "mix": 0.15} # Subtle reverb okay
        },
        "rim":        { # Sparse syncopation, kept minimal
             "note_probability": (0.05, 0.2),
             "velocity_range": (60, 75),
             "timing_randomness_ms": (1, 4),
             "double_hit_probability": 0.0,
             "skip_hit_probability": 0.3,
             "ghost_note_probability": 0.05,
             "ghost_note_velocity_factor": 0.5,
             "pattern_weights": [0.1,0.1,0.1,0.5, 0.1,0.1,0.1,0.1, 0.1,0.1,0.6,0.1, 0.1,0.1,0.1,0.1],
        },
        # Keep other perc sparse/off for default house
        "cowbell":    { "note_probability": (0.0, 0.01),"velocity_range": (90, 100),"timing_randomness_ms": (0, 1),"double_hit_probability": 0.0,"skip_hit_probability": 0.5,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.0]*16,},
        "claves":     { "note_probability": (0.0, 0.0),"velocity_range": (85, 95),"timing_randomness_ms": (0, 0),"double_hit_probability": 0.0,"skip_hit_probability": 0.0,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.0]*16,},
        "low_tom":    { "note_probability": (0.0, 0.05),"velocity_range": (85, 105),"timing_randomness_ms": (1, 5),"double_hit_probability": 0.0,"skip_hit_probability": 0.4,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1, 0.1,0.1,0.4,0.1],},
        "mid_tom":    { "note_probability": (0.0, 0.0),"velocity_range": (90, 110),"timing_randomness_ms": (0, 2),"double_hit_probability": 0.0,"skip_hit_probability": 0.1,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.1]*16,},
        "high_tom":   { "note_probability": (0.0, 0.0),"velocity_range": (90, 110),"timing_randomness_ms": (0, 2),"double_hit_probability": 0.0,"skip_hit_probability": 0.1,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.1]*16,},
        "swing_amount": 0.0, # Default swing is 0, rely on section override
    },
    "jungle": {
        "description": "High-speed, syncopated breakbeat style.",
        "kick": {"note_probability": (0.3, 0.6),"velocity_range": (100, 127),"timing_randomness_ms": (0, 5),"double_hit_probability": 0.1,"skip_hit_probability": 0.15,"ghost_note_probability": 0.1,"ghost_note_velocity_factor": 0.5,"pattern_weights": [0.8, 0.3, 0.1, 0.5, 0.2, 0.6, 0.1, 0.7, 0.8, 0.1, 0.4, 0.2, 0.1, 0.7, 0.3, 0.6],},
        "snare": {"note_probability": (0.2, 0.5),"velocity_range": (90, 120),"timing_randomness_ms": (0, 8),"double_hit_probability": 0.25,"skip_hit_probability": 0.05,"ghost_note_probability": 0.3,"ghost_note_velocity_factor": 0.4,"pattern_weights": [0.1, 0.2, 0.4, 0.7, 0.9, 0.3, 0.5, 0.2, 0.1, 0.6, 0.3, 0.8, 1.0, 0.2, 0.5, 0.4],},
        "closed_hat": {"note_probability": (0.6, 0.9),"velocity_range": (55, 80),"timing_randomness_ms": (0, 4),"double_hit_probability": 0.15,"skip_hit_probability": 0.1,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.8, 0.9, 0.7, 0.9, 0.8, 0.9, 0.7, 0.9, 0.8, 0.9, 0.7, 0.9, 0.8, 0.9, 0.7, 0.9],},
        "open_hat": {"note_probability": (0.1, 0.3),"velocity_range": (65, 90),"timing_randomness_ms": (0, 6),"double_hit_probability": 0.0,"skip_hit_probability": 0.1,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.1, 0.1, 0.4, 0.1, 0.1, 0.5, 0.1, 0.3, 0.1, 0.1, 0.6, 0.1, 0.1, 0.4, 0.1, 0.2],},
        "rim": {
            "note_probability": (0.2, 0.5),
            "velocity_range": (70, 95),
            "timing_randomness_ms": (2, 10),
            "double_hit_probability": 0.15,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.2,
            "ghost_note_velocity_factor": 0.5,
            "pattern_weights": [0.2,0.5,0.8,0.4, 0.6,0.3,0.7,0.2, 0.3,0.8,0.5,0.6, 0.9,0.4,0.7,0.5],
            "delay_params": {"time_ms": 140, "feedback": 0.4, "mix": 0.25},
        },
        "clap": {
            "note_probability": (0.0, 0.05),
            "velocity_range": (80, 100),
            "timing_randomness_ms": (5, 15),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.1]*16,
        },
        "cowbell": {
            "note_probability": (0.0, 0.0),
            "velocity_range": (90, 100),
            "timing_randomness_ms": (0, 5),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.0]*16,
        },
        "claves": {
            "note_probability": (0.0, 0.0),
            "velocity_range": (85, 95),
            "timing_randomness_ms": (0, 2),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.05,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.0]*16,
        },
        "low_tom": {
            "note_probability": (0.05, 0.2),
            "velocity_range": (95, 115),
            "timing_randomness_ms": (1, 6),
            "double_hit_probability": 0.1,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.1,
            "ghost_note_velocity_factor": 0.5,
            "pattern_weights": [0.4,0.1,0.2,0.1, 0.5,0.1,0.3,0.1, 0.4,0.1,0.2,0.1, 0.5,0.1,0.3,0.1],
        },
        "mid_tom": {
            "note_probability": (0.05, 0.2),
            "velocity_range": (95, 115),
            "timing_randomness_ms": (1, 6),
            "double_hit_probability": 0.1,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.1,
            "ghost_note_velocity_factor": 0.5,
            "pattern_weights": [0.1,0.3,0.1,0.4, 0.1,0.5,0.1,0.2, 0.1,0.3,0.1,0.4, 0.1,0.5,0.1,0.2],
        },
        "high_tom": {
            "note_probability": (0.05, 0.2),
            "velocity_range": (95, 115),
            "timing_randomness_ms": (1, 6),
            "double_hit_probability": 0.1,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.1,
            "ghost_note_velocity_factor": 0.5,
            "pattern_weights": [0.1,0.1,0.4,0.1, 0.1,0.1,0.5,0.1, 0.1,0.1,0.4,0.1, 0.1,0.1,0.5,0.1],
        },
        "swing_amount": 0.0, 
    },
    "techno": { 
        "description": "Driving techno groove with four-on-the-floor kick.",
        "use_strict_pattern": True, # <<< ENABLE STRICT PATTERN
        "kick":       {
            "note_probability": (1.0, 1.0), # Overridden by strict
            "velocity_range": (115, 127),   
            "timing_randomness_ms": (0, 0), 
            "double_hit_probability": 0.0,     
            "skip_hit_probability": 0.0,    
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], # Kicks on 1,2,3,4
        },
        "snare":      { # Not used in strict techno if clap active
             "note_probability": (0.0, 0.0),"velocity_range": (80, 100),"timing_randomness_ms": (0, 0),"double_hit_probability": 0.0, "skip_hit_probability": 0.0,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.0]*16,},
         "clap":      { # Strict on 2 & 4
            "note_probability": (1.0, 1.0),"velocity_range": (95, 115),"timing_randomness_ms": (0, 0),"double_hit_probability": 0.0,"skip_hit_probability": 0.0,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],},
        "closed_hat": { # Strict on off-beat 8ths
             "note_probability": (0.8, 0.95),
            "velocity_range": (50, 75), # <<< LOWERED VELOCITY RANGE SIGNIFICANTLY
            "timing_randomness_ms": (0, 2),
            "double_hit_probability": 0.01,
            "skip_hit_probability": 0.05,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.1, 1.0, 0.2, 0.8, 0.1, 1.0, 0.2, 0.8, 0.1, 1.0, 0.2, 0.8, 0.1, 1.0, 0.2, 0.8],
            "delay_params": {"time_ms": 180, "feedback": 0.25, "mix": 0.2}
         },
        "open_hat":   { # Strict on last 16th of bar?
            "note_probability": (0.2, 0.4),
            "velocity_range": (60, 85), # <<< LOWERED VELOCITY RANGE 
            "timing_randomness_ms": (0, 4),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.1, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.9],
            "delay_params": {"time_ms": 240, "feedback": 0.3, "mix": 0.25}
        },
        # Other percussion off by default in strict techno
        "rim":        { "note_probability": (0.0, 0.0),"velocity_range": (70, 90),"timing_randomness_ms": (0, 0),"double_hit_probability": 0.0,"skip_hit_probability": 0.0,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.0]*16,},
        "cowbell":    { "note_probability": (0.0, 0.0),"velocity_range": (90, 100),"timing_randomness_ms": (0, 0),"double_hit_probability": 0.0,"skip_hit_probability": 0.0,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.0]*16,},
        "claves":     { "note_probability": (0.0, 0.0),"velocity_range": (85, 95),"timing_randomness_ms": (0, 0),"double_hit_probability": 0.0,"skip_hit_probability": 0.0,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.0]*16,},
        "low_tom":    { 
            "note_probability": (0.05, 0.1), # <<< Give it a small base probability
            "velocity_range": (90, 110),
            "timing_randomness_ms": (0, 5),  # <<< Allow slight timing variation 
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.1,     # <<< Allow rare skips
            "ghost_note_probability": 0.0,  
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0.4, 0.1, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0.5, 0.1], # <<< Add subtle weights
        },
        "mid_tom":    { "note_probability": (0.0, 0.0),"velocity_range": (90, 110),"timing_randomness_ms": (0, 0),"double_hit_probability": 0.0,"skip_hit_probability": 0.0,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.0]*16,},
        "high_tom":   { "note_probability": (0.0, 0.0),"velocity_range": (90, 110),"timing_randomness_ms": (0, 0),"double_hit_probability": 0.0,"skip_hit_probability": 0.0,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.0]*16,},
        "swing_amount": 0.0, 
    },
    "minimal": { # Sparse, clean minimal
        "description": "Sparse, clean rhythm, often less syncopated.",
        "kick": {"note_probability": (0.4, 0.7),"velocity_range": (100, 115),"timing_randomness_ms": (0, 3),"double_hit_probability": 0.0,"skip_hit_probability": 0.1,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.9, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1],},
        "snare": {"note_probability": (0.1, 0.3),"velocity_range": (75, 95),"timing_randomness_ms": (0, 5),"double_hit_probability": 0.0,"skip_hit_probability": 0.2,"ghost_note_probability": 0.05,"ghost_note_velocity_factor": 0.5,"pattern_weights": [0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1],},
         "clap": {"note_probability": (0.0, 0.1),"velocity_range": (90, 100),"timing_randomness_ms": (0, 2),"double_hit_probability": 0.0,"skip_hit_probability": 0.3,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.1]*16,},
        "closed_hat": {"note_probability": (0.5, 0.8),"velocity_range": (40, 60),"timing_randomness_ms": (0, 3),"double_hit_probability": 0.0,"skip_hit_probability": 0.15,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.6, 0.3, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3],},
        "open_hat": {"note_probability": (0.05, 0.15),"velocity_range": (50, 75),"timing_randomness_ms": (0, 4),"double_hit_probability": 0.0,"skip_hit_probability": 0.2,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.1]*16,},
        "rim": {
            "note_probability": (0.05, 0.2),
            "velocity_range": (65, 85),
            "timing_randomness_ms": (1, 4),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.4,
            "ghost_note_probability": 0.1,
            "ghost_note_velocity_factor": 0.5,
            "pattern_weights": [0.1,0.1,0.6,0.1, 0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1],
            "delay_params": {"time_ms": 350, "feedback": 0.4, "mix": 0.5},
        },
        "cowbell": {"note_probability": (0.0, 0.0),"velocity_range": (90, 100),"timing_randomness_ms": (0, 1),"double_hit_probability": 0.0,"skip_hit_probability": 0.1,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.0]*16,},
        "claves": {"note_probability": (0.05, 0.15),"velocity_range": (80, 90),"timing_randomness_ms": (0, 2),"double_hit_probability": 0.0,"skip_hit_probability": 0.1,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.1,0.1,0.1,0.3, 0.1,0.1,0.1,0.3, 0.1,0.1,0.1,0.3, 0.1,0.1,0.1,0.3],},
        "low_tom": {"note_probability": (0.0, 0.02),"velocity_range": (90, 110),"timing_randomness_ms": (0, 2),"double_hit_probability": 0.0,"skip_hit_probability": 0.1,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.1]*16,},
        "mid_tom": {"note_probability": (0.0, 0.02),"velocity_range": (90, 110),"timing_randomness_ms": (0, 2),"double_hit_probability": 0.0,"skip_hit_probability": 0.1,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.1]*16,},
        "high_tom": {"note_probability": (0.0, 0.02),"velocity_range": (90, 110),"timing_randomness_ms": (0, 2),"double_hit_probability": 0.0,"skip_hit_probability": 0.1,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.1]*16,},
        "swing_amount": 0.0, 
    },
    "ambient": {
        "description": "Sparse, atmospheric textures with subtle pulse.",
        "use_strict_pattern": False, # Allow probabilities/weights to work
        "kick": { # Extremely rare, deep pulse, maybe on the 1 of every 4 bars? Use weights.
            "note_probability": (0.6, 0.8), # High base prob needed because weights are mostly 0
            "velocity_range": (70, 95),      # Soft
            "timing_randomness_ms": (10, 40), # Loose timing
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.75,     # HIGH SKIP RATE
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            # Weight only on the very first step of the 16-step pattern
            "pattern_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "reverb_params": {"time_ms": 1500, "mix": 0.4, "decay": 0.5}
        },
        "snare": { # Very rare, perhaps a syncopated accent
            "note_probability": (0.5, 0.7),
            "velocity_range": (50, 70), # Soft
            "timing_randomness_ms": (15, 50),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.85,     # EXTREMELY HIGH SKIP RATE
            "ghost_note_probability": 0.01,
            "ghost_note_velocity_factor": 0.5,
            # Weight only on the 13th step (e.g., the 'a' of 3)
            "pattern_weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "delay_params": {"time_ms": 700, "feedback": 0.6, "mix": 0.5},
            "reverb_params": {"time_ms": 1500, "mix": 0.5, "decay": 0.6}
        },
        "clap": { # Extremely rare
            "note_probability": (0.0, 0.01),
            "velocity_range": (70, 90),
            "timing_randomness_ms": (5, 20),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.95,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.1]*16, # Flat low weight
            "reverb_params": {"time_ms": 1200, "mix": 0.4, "decay": 0.55}
        },
        "closed_hat": { # Sparse pulse on off-beats
            "note_probability": (0.4, 0.6),
            "velocity_range": (30, 50), # Very soft
            "timing_randomness_ms": (10, 30),
            "double_hit_probability": 0.0, 
            "skip_hit_probability": 0.6, # High skip rate
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5], # Weights on '+' counts, stronger on 1+/3+
            "delay_params": {"time_ms": 400, "feedback": 0.4, "mix": 0.45},
            "reverb_params": {"time_ms": 800, "mix": 0.4, "decay": 0.65}
        },
        "open_hat": { # Rare atmospheric wash
            "note_probability": (0.1, 0.3),
            "velocity_range": (40, 65),
            "timing_randomness_ms": (15, 50),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.7,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0], # Weight on last 8th of beats 2 & 4
            "delay_params": {"time_ms": 800, "feedback": 0.5, "mix": 0.5},
            "reverb_params": {"time_ms": 1800, "mix": 0.6, "decay": 0.7} 
        },
         "ride_cymbal": { # Slow pulse, maybe quarter notes
            "note_probability": (0.3, 0.6),
            "velocity_range": (50, 70),
            "timing_randomness_ms": (5, 20),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.5,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1], # Weight on downbeats
            "delay_params": {"time_ms": 700, "feedback": 0.65, "mix": 0.5},
            "reverb_params": {"time_ms": 3000, "mix": 0.7, "decay": 0.6} 
        },
        "crash_cymbal": { # Extremely rare 
            "note_probability": (0.0, 0.01),
            "velocity_range": (60, 80),
            "timing_randomness_ms": (10, 50),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.6,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "reverb_params": {"time_ms": 5000, "mix": 0.8, "decay": 0.7} 
        },
        "rim":        { # Sparse texture, very high skip
            "note_probability": (0.1, 0.3),
            "velocity_range": (40, 65),
            "timing_randomness_ms": (15, 40),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.8, # Higher skip
            "ghost_note_probability": 0.1,
            "ghost_note_velocity_factor": 0.5, 
            "pattern_weights": [0.1, 0.4, 0.1, 0.6, 0.1, 0.4, 0.1, 0.6, 0.1, 0.4, 0.1, 0.6, 0.1, 0.4, 0.1, 0.6],
            "delay_params": {"time_ms": 600, "feedback": 0.4, "mix": 0.6}, 
            "reverb_params": {"time_ms": 1000, "mix": 0.4, "decay": 0.6}
         },
        "cowbell":    { "note_probability": (0.0, 0.01),"velocity_range": (80, 95),"timing_randomness_ms": (5, 15),"double_hit_probability": 0.0,"skip_hit_probability": 0.9,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0, "pattern_weights": [0.1]*16,"delay_params": {"time_ms": 400, "feedback": 0.2, "mix": 0.3},"reverb_params": {"time_ms": 200, "mix": 0.2}},
        "claves":     { "note_probability": (0.01, 0.05),"velocity_range": (75, 90),"timing_randomness_ms": (2, 10),"double_hit_probability": 0.0,"skip_hit_probability": 0.8,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0, "pattern_weights": [0.1,0.1,0.1,0.8, 0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.8, 0.1,0.1,0.1,0.1],"delay_params": {"time_ms": 100, "feedback": 0.1, "mix": 0.6}, "reverb_params": {"time_ms": 200, "mix": 0.2}},
        "low_tom":    { "note_probability": (0.01, 0.05),"velocity_range": (70, 95),"timing_randomness_ms": (10, 30),"double_hit_probability": 0.0,"skip_hit_probability": 0.7,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], "reverb_params": {"time_ms": 1200, "mix": 0.4, "decay": 0.5}},
        "mid_tom":    { "note_probability": (0.01, 0.04),"velocity_range": (70, 95),"timing_randomness_ms": (10, 30),"double_hit_probability": 0.0,"skip_hit_probability": 0.8,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1], "reverb_params": {"time_ms": 1000, "mix": 0.35, "decay": 0.55}},
        "high_tom":   { "note_probability": (0.01, 0.03),"velocity_range": (70, 95),"timing_randomness_ms": (10, 30),"double_hit_probability": 0.0,"skip_hit_probability": 0.9,"ghost_note_probability": 0.0,"ghost_note_velocity_factor": 0.0,"pattern_weights": [0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.4], "reverb_params": {"time_ms": 800, "mix": 0.3, "decay": 0.6}},
        "swing_amount": 0.0, 
    },
    # --- NEW PRESET: deep_lofi (Revised for Chill Vibe) ---
    "deep_lofi": {
        "description": "Chill, atmospheric lofi beat with smooth textures.",
        "use_strict_pattern": False,
        "kick": { 
            "note_probability": (0.4, 0.6), # Higher base, rely on weights/skips
            "velocity_range": (60, 85),      # Soft 
            "timing_randomness_ms": (3, 15), # Reduced randomness
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.15,    # Moderate skips
            "ghost_note_probability": 0.05,   
            "ghost_note_velocity_factor": 0.4,
            # Weights: Focus on 1, sometimes 'a' of 2 or 4, less complex
            "pattern_weights": [1.0, 0.1, 0.1, 0.2, 0.1, 0.1, 0.5, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1], 
            # Removed default reverb/delay
        },
        "snare": { # Very sparse, mostly ghosts if present
            "note_probability": (0.05, 0.15), 
            "velocity_range": (50, 70), 
            "timing_randomness_ms": (5, 20),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.7, # High skip, prefer rim     
            "ghost_note_probability": 0.2, # More likely ghost than full snare 
            "ghost_note_velocity_factor": 0.5,
            "pattern_weights": [0.1]*16, # Flat low weight
        },
        "rim": { # Primary backbeat, simple pattern
            "note_probability": (0.6, 0.85),
            "velocity_range": (55, 75), # Soft rim
            "timing_randomness_ms": (4, 18),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.1,     
            "ghost_note_probability": 0.1, # Allow ghost notes on off beats
            "ghost_note_velocity_factor": 0.6,
            # Weights: Mainly 2 & 4, possibility of ghost on off-beats
            "pattern_weights": [0.1, 0.1, 0.1, 0.1, 1.0, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.2, 0.1, 0.1], 
        },
        "closed_hat": { 
            "note_probability": (0.5, 0.8),
            "velocity_range": (30, 55), # Very soft
            "timing_randomness_ms": (2, 12), # Tighter timing
            "double_hit_probability": 0.0, 
            "skip_hit_probability": 0.15, 
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            # Weights: Consistent 8ths or 16ths feel, slightly favour off-beats
            "pattern_weights": [0.7, 0.9, 0.7, 0.8, 0.7, 0.9, 0.7, 0.8, 0.7, 0.9, 0.7, 0.8, 0.7, 0.9, 0.7, 0.8], 
        },
        "open_hat": { # Atmospheric, less frequent
            "note_probability": (0.05, 0.15),
            "velocity_range": (40, 65),
            "timing_randomness_ms": (8, 25),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.5, # More likely to skip
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            # Weights: Infrequent, maybe last beat
            "pattern_weights": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8], 
            # Add subtle reverb for atmosphere
            "reverb_params": {"time_ms": 800, "mix": 0.3, "decay": 0.6} 
        },
        "claves": { # Very sparse texture 
            "note_probability": (0.01, 0.05), 
            "velocity_range": (55, 75),
            "timing_randomness_ms": (3, 10),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.7,
            "ghost_note_probability": 0.05,
            "ghost_note_velocity_factor": 0.4, 
            "pattern_weights": [0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.6, 0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.4],
        },
        "low_tom": { # Minimal deep element
            "note_probability": (0.01, 0.05), 
            "velocity_range": (50, 80),
            "timing_randomness_ms": (15, 35),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.6,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            "pattern_weights": [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
        },
        # Omit low_conga, maracas by default for cleaner sound
        "swing_amount": 0.55, # Solid lofi swing
    },
    # --- END REVISED PRESET ---
    # --- NEW PRESET: classic_hip_hop (Revised) ---
    "classic_hip_hop": {
        "description": "Boom bap style hip hop groove with strong kick/snare and swing.",
        "use_strict_pattern": False,
        "kick": {
            "note_probability": (0.6, 0.85), 
            "velocity_range": (95, 120),      
            "timing_randomness_ms": (1, 8), # Reduced randomness range
            "double_hit_probability": 0.01, # Less likely double hits
            "skip_hit_probability": 0.05, # Less likely skips     
            "ghost_note_probability": 0.02, # Fewer ghosts 
            "ghost_note_velocity_factor": 0.4,
            # Weights: More defined kick pattern
            "pattern_weights": [1.0, 0.5, 0.1, 0.2, 0.3, 0.1, 0.6, 0.1, 0.9, 0.1, 0.1, 0.1, 0.4, 0.1, 0.7, 0.1], 
        },
        "snare": { # Strong backbeat
            "note_probability": (0.8, 0.95), # Higher probability for snare
            "velocity_range": (100, 127), 
            "timing_randomness_ms": (1, 6), # Tighter timing for snare
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.02, # Very low skip on snare     
            "ghost_note_probability": 0.05, # Reduced ghosts
            "ghost_note_velocity_factor": 0.5,
            # Weights: Strongly on 2 & 4
            "pattern_weights": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
        },
        "rim": { # Used very sparsely, if at all
            "note_probability": (0.05, 0.2),
            "velocity_range": (70, 90), 
            "timing_randomness_ms": (3, 10),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.3,     
            "ghost_note_probability": 0.05,
            "ghost_note_velocity_factor": 0.6,
            "pattern_weights": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], # Same as snare, lower prob
        },
        "closed_hat": { # Standard 8ths or 16ths
            "note_probability": (0.7, 0.9), # High probability for hats
            "velocity_range": (55, 80), # Slightly louder default hats
            "timing_randomness_ms": (1, 6), # Tight timing
            "double_hit_probability": 0.01, # Very few doubles
            "skip_hit_probability": 0.05, 
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            # Weights: Consistent 8ths pattern
            "pattern_weights": [0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0], 
        },
        "open_hat": { # Occasional accent
            "note_probability": (0.02, 0.1),
            "velocity_range": (65, 95),
            "timing_randomness_ms": (2, 8),
            "double_hit_probability": 0.0,
            "skip_hit_probability": 0.1,
            "ghost_note_probability": 0.0,
            "ghost_note_velocity_factor": 0.0,
            # Weights: Only on the last 16th
            "pattern_weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
        },
        # Omit other perc like toms, claves, cowbell for classic feel
        "swing_amount": 0.55, # Keep strong swing
    },
    # --- END REVISED PRESET ---
}

# --- General MIDI Drum Map (Common) ---
GM_DRUM_NOTES = {
    "kick": 36,
    "snare": 38,
    "closed_hat": 42,
    "open_hat": 46,
    "crash_cymbal": 49,
    "ride_cymbal": 51,
    "clap": 39,
    # --- NEW PERCUSSION ---
    "rim": 37,          # Often used instead of/with snare
    "cowbell": 56,
    "claves": 75,
    "low_tom": 41,      # Adding some basic toms
    "mid_tom": 47,
    "high_tom": 50,
    # Add more as needed (e.g., shaker, tambourine, etc.)
}

# --- Helper Functions ---
def bpm_to_tempo(bpm):
    """Converts BPM to microseconds per beat for MIDI tempo meta message."""
    return int(60_000_000 / bpm)

def ms_to_ticks(ms, bpm, ticks_per_beat):
    """Converts milliseconds to MIDI ticks."""
    if bpm == 0: return 0 # Avoid division by zero
    seconds_per_beat = 60.0 / bpm
    ticks_per_second = ticks_per_beat / seconds_per_beat
    return int(round(ms / 1000.0 * ticks_per_second))

# --- HELPER: Note Placement Logic (Uses Pattern Weights) ---
def _should_place_note(style_preset, drum_preset, drum_type, beat, step, only_kicks, density_factor,
                         # Deep Percussion Evolution Params
                         current_abs_tick=0,
                         total_ticks=1,
                         deep_perc_instruments=None,
                         deep_perc_prob_start=0.1,
                         deep_perc_prob_end=0.5,
                         # General Evolution Params
                         evolve_instruments=None,
                         evolve_param="probability",
                         evolve_start=0.0,
                         evolve_end=1.0,
                         # Adherence Param
                         adherence=0.0
                         ):
    """Determines note placement probability, considering evolution and adherence."""
    
    # --- Calculate Base Probability (before general evolution & adherence scaling) ---
    base_prob = 0.0
    # Deep Percussion Evolution multiplier
    deep_evolution_prob_multiplier = 1.0
    is_deep_perc = deep_perc_instruments is not None and drum_type in deep_perc_instruments
    if is_deep_perc and total_ticks > 0:
        progress = float(current_abs_tick) / float(total_ticks)
        deep_evolution_prob_multiplier = deep_perc_prob_start + (deep_perc_prob_end - deep_perc_prob_start) * progress

    use_strict = style_preset.get("use_strict_pattern", False)
    if not drum_preset: return False
    
    if use_strict:
        # Strict rules determine base probability (0 or 1)
        if drum_type == "kick": base_prob = 1.0 if step == 0 else 0.0
        elif drum_type == "clap": base_prob = 1.0 if (step == 0 and (beat % 2 == 1)) else 0.0
        elif drum_type == "closed_hat": base_prob = 1.0 if (step == 1 or step == 3) else 0.0
        elif drum_type == "open_hat": base_prob = 1.0 if (step == 2 and beat % 2 == 1) else 0.0
        elif drum_type == "snare": base_prob = 0.0
    else:
        # Probabilistic calculation
        note_prob_min, note_prob_max = drum_preset["note_probability"]
        calculated_prob = note_prob_min + (note_prob_max - note_prob_min) * density_factor
        pattern_weights = drum_preset.get("pattern_weights", [1.0] * 16)
        current_step_in_bar = (beat * 4 + step) % 16
        step_weight = pattern_weights[current_step_in_bar]
        base_prob = calculated_prob * step_weight

        # Apply deep evolution multiplier to the base probability
        base_prob *= deep_evolution_prob_multiplier
        
        if only_kicks and drum_type == "kick": 
            # Special override for only_kicks scenario (simplified)
            if step == 0: base_prob = 1.0
            elif step == 2 and beat % 2 == 1: base_prob = max(base_prob, 0.5 * deep_evolution_prob_multiplier)
            elif step in [1, 3]: base_prob = max(base_prob, 0.2 * deep_evolution_prob_multiplier)
            else: base_prob = 0.0 # Ensure kick doesn't randomly appear elsewhere
            
    # --- General Evolution (Probability) ---
    prob_after_general_evo = base_prob # Start with the calculated base probability
    is_general_evolve_instrument = evolve_instruments is not None and drum_type in evolve_instruments
    if is_general_evolve_instrument and evolve_param == "probability" and total_ticks > 0:
        progress = float(current_abs_tick) / float(total_ticks)
        general_evolve_multiplier = evolve_start + (evolve_end - evolve_start) * progress
        prob_after_general_evo *= general_evolve_multiplier # Apply general evolution scaling
        prob_after_general_evo = max(0.0, min(1.0, prob_after_general_evo)) 
    # --- End General Evolution ---
    
    # --- Adherence Logic --- 
    final_probability = prob_after_general_evo # Start with potentially evolved probability
    # Apply adherence only to core rhythm elements and if adherence > 0
    core_elements = ["kick", "snare", "rim"]
    if adherence > 0 and drum_type in core_elements and not use_strict:
        pattern_weights = drum_preset.get("pattern_weights", [1.0] * 16)
        current_step_in_bar = (beat * 4 + step) % 16
        step_weight = pattern_weights[current_step_in_bar]
        # Determine intended placement based on weight threshold
        intended_placement = (step_weight > 0.5) 
        
        if intended_placement:
            # Boost probability towards 1.0 based on adherence
            final_probability = prob_after_general_evo + (1.0 - prob_after_general_evo) * adherence
        else:
            # Reduce probability towards 0.0 based on adherence
            final_probability = prob_after_general_evo * (1.0 - adherence)
            
        # Ensure probability stays within bounds after adherence adjustment
        final_probability = max(0.0, min(1.0, final_probability))
    # --- End Adherence ---

    # Final random check
    place_note = random.random() < final_probability
    return place_note

# --- HELPER: Probabilistic Variations (Respects Lead Element More Strongly) ---
def _apply_variations(place_note, drum_preset, is_lead_element, verbosity=0, beat=0, step=0, drum_type="",
                         predictability=0.5):
    """Applies skip and ghost note probabilities, scaled by predictability.
       NEVER skips the lead element. Ghost notes only apply to non-lead.
    """
    is_ghost = False
    place_final = place_note # Start with the initial decision
    variation_factor = 1.0 - predictability # 0.0 = full variation, 1.0 = no variation

    if not is_lead_element:
        skip_prob = drum_preset.get("skip_hit_probability", 0.0) * variation_factor
        ghost_prob = drum_preset.get("ghost_note_probability", 0.0) * variation_factor
        
        # Skip hit? (Only apply if NOT the designated lead element)
        if place_final and random.random() < skip_prob:
            place_final = False
            if verbosity >= 2: print(f"[DEBUG] {drum_type}: Skipped hit at beat {beat+1}.{step+1} (Non-Lead, Predictability={predictability:.2f})")
        
        # Ghost note? (Only if the spot is currently empty AND NOT lead)
        if not place_final and random.random() < ghost_prob:
            place_final = True
            is_ghost = True
            if verbosity >= 2: print(f"[DEBUG] {drum_type}: Added ghost note at beat {beat+1}.{step+1} (Predictability={predictability:.2f})")
    
    # Ensure lead element always plays if its motif said so
    elif is_lead_element and place_note: 
        place_final = True
        is_ghost = False # Lead elements cannot be ghosts
        
    return place_final, is_ghost

# --- HELPER: Timing Calculation (Accepts section_swing) ---
def _calculate_final_timing(abs_tick, section_swing, drum_preset, is_off_beat_16th, bpm, TPB,
                            # --- NEW: Evolution Params ---
                            drum_type="",
                            current_abs_tick=0,
                            total_ticks=1,
                            deep_perc_instruments=None,
                            deep_perc_timing_ms_start=15.0,
                            deep_perc_timing_ms_end=3.0,
                            # --- NEW: Predictability Param ---
                            predictability=0.5
                            # --- END NEW ---
                            ):
    note_time = float(abs_tick) 
    ticks_per_16th = float(TPB / 4.0)
    if section_swing > 0 and is_off_beat_16th:
        swing_delay_ticks = ticks_per_16th * section_swing * (2.0/3.0)
        note_time += swing_delay_ticks

    # Calculate variation factor based on predictability (0.0 = max variation, 1.0 = no variation)
    variation_factor = 1.0 - predictability
    apply_timing_randomness = random.random() < variation_factor # Decide IF randomness is applied
    
    random_offset_ticks = 0
    if apply_timing_randomness:
        # --- Deep Percussion Evolution for Timing ---
        is_deep_perc = deep_perc_instruments is not None and drum_type in deep_perc_instruments
        timing_dev_min, timing_dev_max = drum_preset.get("timing_randomness_ms", (0, 0))
        
        # Scale timing deviation range by predictability
        scaled_timing_dev_min = timing_dev_min # Keep min as is? Or scale too?
        scaled_timing_dev_max = timing_dev_max * variation_factor # Scale only the max deviation
        if scaled_timing_dev_min > scaled_timing_dev_max: scaled_timing_dev_min = scaled_timing_dev_max # Ensure min <= max
        
        current_timing_deviation_ms = 0
        if is_deep_perc and total_ticks > 0:
            progress = float(current_abs_tick) / float(total_ticks)
            evolved_timing_ms = deep_perc_timing_ms_start + (deep_perc_timing_ms_end - deep_perc_timing_ms_start) * progress
            # Apply the evolved timing as a range around 0, scaled by predictability
            evolved_timing_range = evolved_timing_ms * variation_factor
            current_timing_deviation_ms = random.uniform(-evolved_timing_range / 2.0, evolved_timing_range / 2.0)
        else:
            # Use standard timing deviation range from preset, scaled by predictability
            current_timing_deviation_ms = random.uniform(scaled_timing_dev_min, scaled_timing_dev_max)
        # --- End Evolution Logic ---

        if current_timing_deviation_ms != 0:
            random_offset_ticks = ms_to_ticks(current_timing_deviation_ms, bpm, TPB)
    
    note_time += random_offset_ticks # Apply the (potentially zero) offset
    return max(0.0, note_time)

# --- HELPER: Velocity Calculation (with Refined Accenting) ---
def _calculate_final_velocity(drum_preset, is_ghost, step, beat, drum_type="",
                                # Deep Percussion Evolution Params
                                current_abs_tick=0,
                                total_ticks=1,
                                deep_perc_instruments=None,
                                deep_perc_vel_start=70,
                                deep_perc_vel_end=110,
                                # Predictability Param
                                predictability=0.5,
                                # General Evolution Params
                                evolve_instruments=None,
                                evolve_param="probability",
                                evolve_start=0.0,
                                evolve_end=1.0
                                ):
    vel_min, vel_max = drum_preset.get("velocity_range", (80, 100))
    variation_factor = 1.0 - predictability
    apply_velocity_randomness = random.random() < variation_factor # Decide IF velocity randomness is applied

    # --- Deep Percussion Evolution for Velocity ---
    is_deep_perc = deep_perc_instruments is not None and drum_type in deep_perc_instruments
    if is_deep_perc and total_ticks > 0:
        progress = float(current_abs_tick) / float(total_ticks)
        evolved_vel_min = int(deep_perc_vel_start + (deep_perc_vel_end - deep_perc_vel_start) * progress * 0.8) 
        evolved_vel_max = int(deep_perc_vel_start + (deep_perc_vel_end - deep_perc_vel_start) * progress)
        vel_min = max(1, min(127, evolved_vel_min))
        vel_max = max(1, min(127, evolved_vel_max))
        if vel_min > vel_max: vel_min = vel_max
    # --- End Evolution ---
    
    # --- Determine Base Velocity (Potentially link to weights later) ---
    # For now, use midpoint as a target before random variation
    target_velocity = vel_min + (vel_max - vel_min) * 0.75 
    
    velocity = target_velocity # Start with target
    if apply_velocity_randomness:
        # Scale Velocity Randomness Range by Predictability 
        vel_range_random_part = (vel_max - vel_min) * 0.75 # Range for randomness
        scaled_vel_range_random_part = vel_range_random_part * variation_factor
        # Apply randomness around the target velocity
        velocity = random.randint(max(vel_min, int(target_velocity - scaled_vel_range_random_part / 2)), 
                                  min(vel_max, int(target_velocity + scaled_vel_range_random_part / 2)))
                                  
    base_velocity = int(round(velocity)) # Use the potentially randomized velocity

    if not is_ghost:
        accent_boost = 0
        # Apply accents ON TOP of the base velocity
        if drum_type == "kick" and step == 0: accent_boost = int((vel_max - vel_min) * 0.25)
        elif drum_type in ["clap", "snare"] and step == 0 and (beat % 2 == 1): accent_boost = int((vel_max - vel_min) * 0.20)
        velocity = min(vel_max, base_velocity + accent_boost)
    if is_ghost:
        ghost_factor = drum_preset.get("ghost_note_velocity_factor", 0.5)
        # Apply ghost factor to the calculated base velocity
        velocity = int(base_velocity * ghost_factor) 
        velocity = max(1, velocity)
    
    # Apply final velocity clamping (after ghost note reduction)
    velocity = max(vel_min, min(vel_max, velocity))
    
    # --- General Evolution (Velocity) ---
    final_velocity = velocity # Start with calculated velocity
    is_general_evolve_instrument = evolve_instruments is not None and drum_type in evolve_instruments
    if is_general_evolve_instrument and evolve_param == "velocity" and total_ticks > 0:
        progress = float(current_abs_tick) / float(total_ticks)
        general_evolve_multiplier = evolve_start + (evolve_end - evolve_start) * progress
        final_velocity = int(round(velocity * general_evolve_multiplier))
        # Ensure velocity stays within MIDI bounds (1-127)
        final_velocity = max(1, min(127, final_velocity)) 
    # --- End General Evolution ---
    
    return final_velocity

# --- Core Generation Logic (Stronger Lead Motif Generation) ---
def generate_multi_drum_groove(style="dusty", bpm=90, 
                               structure=None, 
                               default_included_drums=None, 
                               verbosity=1,
                               phrase_length_bars=4,
                               # --- NEW: Deep Percussion Evolution Params ---
                               deep_perc_instruments=None, 
                               deep_perc_prob_start=0.1, 
                               deep_perc_prob_end=0.5, 
                               deep_perc_vel_start=70, 
                               deep_perc_vel_end=110, 
                               deep_perc_timing_ms_start=15.0, 
                               deep_perc_timing_ms_end=3.0,
                               # --- NEW: Predictability Param ---
                               predictability=0.5,
                               # --- NEW: General Evolution Params ---
                               evolve_instruments=None,
                               evolve_param="probability",
                               evolve_start=0.0,
                               evolve_end=1.0,
                               # --- NEW: Adherence Param ---
                               adherence=0.0
                               # --- END NEW ---
                               ): 
    """Generates a MIDI groove using phrase-based motif repetition with stronger lead."""
    
    # --- Setup --- (Mostly unchanged)
    if style not in STYLE_PRESETS: style = "dusty"
    style_preset = STYLE_PRESETS[style]
    TPB = 480
    mid = MidiFile(ticks_per_beat=TPB)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=bpm_to_tempo(bpm), time=0))
    track.append(MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    ticks_per_16th = TPB // 4
    density_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
    scheduled_notes = []
    current_beat_offset = 0
    total_length_beats = sum(sec["length"] for sec in structure)
    total_ticks = total_length_beats * TPB
    phrase_length_steps = phrase_length_bars * 16 # 16 steps per bar

    print(f"🥁 Generating groove: Style='{style}', BPM={bpm}, Structure Defined ({len(structure)} sections, {total_length_beats} beats total), Phrase Bars={phrase_length_bars}")

    # --- Main Generation Loop (Iterate through Structure) --- 
    for section_index, section in enumerate(structure):
        section_length = section["length"]
        section_density_name = section["density"]
        section_instruments = section["instruments"]
        section_density_factor = density_map.get(section_density_name.lower(), 0.5)
        section_swing = section["swing"]
        section_apply_variations = section["apply_variations"]
        section_lead_instrument = section.get("lead_instrument") # Use .get for safety
        
        instrument_motifs = {} # Store generated motifs for this section

        if verbosity >= 1:
            print(f"--> Section {section_index+1}: Len={section_length}, Dens={section_density_name}, Swing={section_swing:.2f}, Vary={section_apply_variations}, Strict={section_apply_variations}, Lead={section_lead_instrument or 'None'}, Drums=[{', '.join(section_instruments)}]")

        section_active_drums = { drum_type: GM_DRUM_NOTES[drum_type] for drum_type in section_instruments if drum_type in GM_DRUM_NOTES and drum_type in style_preset }
        only_kicks_section = (len(section_active_drums) == 1 and "kick" in section_active_drums)
        
        # --- Generate/Update Motifs at Phrase Start --- 
        for beat_in_section in range(section_length):
            global_beat = current_beat_offset + beat_in_section
            is_phrase_start = (beat_in_section % (phrase_length_bars * 4) == 0)
            if is_phrase_start:
                if verbosity >= 2: print(f"    --- Generating Motifs for Phrase starting at Beat {global_beat+1} ---")
                for drum_type, midi_note in section_active_drums.items():
                    is_current_lead = (drum_type == section_lead_instrument)
                    drum_preset = style_preset[drum_type]
                    motif = []
                    for step_in_phrase in range(phrase_length_steps):
                        pseudo_beat = step_in_phrase // 4
                        pseudo_step = step_in_phrase % 4
                        
                        # --- MODIFIED MOTIF GENERATION FOR LEAD ---
                        if is_current_lead:
                            # For lead, place note if weight > threshold (ignore base prob)
                            pattern_weights = drum_preset.get("pattern_weights", [0.5] * 16) # Need some weight
                            step_weight = pattern_weights[step_in_phrase % 16]
                            should_place_in_motif = (step_weight > 0.5) # Threshold for lead presence
                        else:
                            # For non-lead, use standard probability check
                            # Pass all relevant params, including adherence
                            should_place_in_motif = _should_place_note(style_preset, drum_preset, drum_type, 
                                                                   pseudo_beat, pseudo_step, 
                                                                   only_kicks_section, section_density_factor,
                                                                   # Deep Percussion Params
                                                                   current_abs_tick=global_beat, # Use global beat for context
                                                                   total_ticks=total_ticks,
                                                                   deep_perc_instruments=deep_perc_instruments,
                                                                   deep_perc_prob_start=deep_perc_prob_start,
                                                                   deep_perc_prob_end=deep_perc_prob_end,
                                                                   # General Evolution Params
                                                                   evolve_instruments=evolve_instruments,
                                                                   evolve_param=evolve_param,
                                                                   evolve_start=evolve_start,
                                                                   evolve_end=evolve_end,
                                                                   # Adherence Param
                                                                   adherence=adherence 
                                                                   )
                        # --- END MODIFICATION --- 
                        motif.append(should_place_in_motif)
                        
                    instrument_motifs[drum_type] = motif
                    if verbosity >= 2: print(f"      Motif for {drum_type}{'(LEAD)' if is_current_lead else ''}: {['X' if m else '.' for m in motif]}")
            
            # --- Process each step within the current beat using motifs ---
            for step in range(4):
                abs_tick = (global_beat * 4 + step) * ticks_per_16th
                is_off_beat_16th = (step % 2 != 0)
                step_in_phrase = (beat_in_section * 4 + step) % phrase_length_steps
                
                # Determine strictness for THIS step based on section and preset
                use_strict = style_preset.get("use_strict_pattern", False) and not section_apply_variations

                for drum_type, midi_note in section_active_drums.items():
                    if drum_type not in instrument_motifs: continue 
                    motif = instrument_motifs[drum_type]
                    if not motif[step_in_phrase]: continue 
                    
                    drum_preset = style_preset[drum_type]
                    should_place_from_motif = True
                    is_lead = (drum_type == section_lead_instrument)
                    
                    place_final, is_ghost = (should_place_from_motif, False)
                    if section_apply_variations: 
                        place_final, is_ghost = _apply_variations(should_place_from_motif, drum_preset, is_lead, verbosity, global_beat, step, drum_type, predictability)

                    if place_final:
                        # --- Calculate note_time with potential evolution for deep perc timing --- 
                        evolved_note_time = float(abs_tick)
                        if section_apply_variations:
                            evolved_note_time = _calculate_final_timing(abs_tick, section_swing, drum_preset, is_off_beat_16th, bpm, TPB,
                                                                # Pass evolution params for timing calculation
                                                                drum_type=drum_type, 
                                                                current_abs_tick=abs_tick, 
                                                                total_ticks=total_ticks, 
                                                                deep_perc_instruments=deep_perc_instruments, 
                                                                deep_perc_timing_ms_start=deep_perc_timing_ms_start,
                                                                deep_perc_timing_ms_end=deep_perc_timing_ms_end,
                                                                predictability=predictability
                                                                )
                        
                        # --- Calculate velocity with potential evolution for deep perc velocity & general velocity evolution ---
                        velocity = _calculate_final_velocity(drum_preset, is_ghost, step, beat=global_beat, drum_type=drum_type,
                                                            # Deep Percussion Params
                                                            current_abs_tick=abs_tick,
                                                            total_ticks=total_ticks,
                                                            deep_perc_instruments=deep_perc_instruments,
                                                            deep_perc_vel_start=deep_perc_vel_start,
                                                            deep_perc_vel_end=deep_perc_vel_end,
                                                            # Predictability Param
                                                            predictability=predictability,
                                                            # General Evolution Params
                                                            evolve_instruments=evolve_instruments,
                                                            evolve_param=evolve_param,
                                                            evolve_start=evolve_start,
                                                            evolve_end=evolve_end
                                                            )
                        if use_strict: # Strict velocity overrides evolution
                            velocity = drum_preset["velocity_range"][1]

                        note_off_time = evolved_note_time + (ticks_per_16th / 2.0)
                        if note_off_time >= total_ticks: note_off_time = float(total_ticks - 1)
                        if evolved_note_time >= note_off_time: continue 
                        scheduled_notes.append((evolved_note_time, midi_note, velocity, note_off_time))
                        
                        # --- Double Hit Logic (Apply after main note scheduling) --- 
                        # Scale double hit probability by predictability
                        variation_factor = 1.0 - predictability
                        double_hit_prob = drum_preset.get("double_hit_probability", 0.0) * variation_factor
                        
                        if section_apply_variations and not is_ghost and random.random() < double_hit_prob:
                           # Double hits inherit main note timing/velocity rules (including evolution if applicable)
                           double_hit_offset_ticks = random.randint(ticks_per_16th // 4, ticks_per_16th // 2)
                           # Apply timing randomness/swing to the offset base time
                           base_double_hit_time_abs = evolved_note_time 
                           double_hit_time = base_double_hit_time_abs + double_hit_offset_ticks 
                           # Apply evolution etc. to double hit velocity?
                           double_hit_velocity = max(1, int(velocity * 0.7)) 
                           double_hit_off_time = double_hit_time + (ticks_per_16th / 2.0)
                           if double_hit_time < total_ticks and double_hit_off_time < total_ticks:
                               scheduled_notes.append((double_hit_time, midi_note, double_hit_velocity, double_hit_off_time))
                               if verbosity >= 2: print(f"[DEBUG] {drum_type}: Added double hit at beat {global_beat+1}.{step+1}")
                               
        current_beat_offset += section_length 

    # --- Final MIDI Track Assembly --- 
    scheduled_notes.sort(key=lambda x: x[0]) 
    note_events = []
    for note_on_time, note_num, velocity, note_off_time in scheduled_notes:
        note_events.append((note_on_time, 'note_on', note_num, velocity))
        note_events.append((note_off_time, 'note_off', note_num, 64)) 
    note_events.sort(key=lambda x: x[0])
    last_time_ticks = 0.0 # Use float for tracking last event time accurately
    for abs_time_float, event_type, note_num, velocity in note_events:
        delta_time = max(0, int(round(abs_time_float - last_time_ticks))) # Calculate delta from float time
        track.append(Message(event_type, note=note_num, velocity=velocity, time=delta_time))
        last_time_ticks = abs_time_float # Store precise time for next delta calculation
    end_delta = max(0, int(round(float(total_ticks) - last_time_ticks)))
    track.append(MetaMessage('end_of_track', time=end_delta))
    if verbosity >= 1: print(f"✅ Groove generated with {len(scheduled_notes)} notes for {total_length_beats} beats.")
    return mid

# --- Main Execution --- (Updated call site)
def main():
    parser = argparse.ArgumentParser(description="Multi-Drum Groove Generator with Structure Support")
    parser.add_argument("--style", type=str, default="dusty", help=f"Groove style keyword (e.g., {', '.join(STYLE_PRESETS.keys())})")
    parser.add_argument("--bpm", type=int, default=120, help="Beats per minute (tempo)")
    # Density and Length are now controlled by structure OR used as fallback
    parser.add_argument("--density", type=str, default="medium", choices=["low", "medium", "high"], help="Default groove density if --structure is not used")
    parser.add_argument("--length", type=int, default=16, choices=[16, 32, 64, 128], help="Default length in beats if --structure is not used")
    parser.add_argument("--structure", type=str, default=None, 
                        help="Define beat structure: SECTIONS separated by '/'. Each section: 'LENGTH*DENSITY*INSTRUMENTS[;options]' (e.g., \"16*low*kick+hat/32*medium*default\"). Instruments: '+' separated list, or 'default', or 'all'.")
    
    # Sound inclusion flags (used for 'default' instruments in structure or if no structure given)
    sound_group = parser.add_argument_group('Default Sound Options')
    sound_group.add_argument("--include-kick", action="store_true", default=True, help="Include kick drum (default: True)")
    sound_group.add_argument("--no-kick", action="store_false", dest="include_kick", help="Exclude kick drum")
    sound_group.add_argument("--include-snare", action="store_true", default=True, help="Include snare drum (default: True)")
    sound_group.add_argument("--no-snare", action="store_false", dest="include_snare", help="Exclude snare drum")
    sound_group.add_argument("--include-hat", action="store_true", default=True, help="Include hi-hats (default: True)")
    sound_group.add_argument("--no-hat", action="store_false", dest="include_hat", help="Exclude hi-hats")
    sound_group.add_argument("--include-cymbal", action="store_true", default=False, help="Include cymbals (default: False)")
    sound_group.add_argument("--no-cymbal", action="store_false", dest="include_cymbal", help="Exclude cymbals")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the output file(s)")
    parser.add_argument("--filename", type=str, default="", help="Custom base filename (optional)")
    parser.add_argument("--format", type=str, default="mid", choices=["mid", "wav", "both"], help="Output format (mid, wav, or both)")
    # Updated kit selection argument
    parser.add_argument("--kit", type=str, default="tr-909", choices=["tr-909", "tr-808"], 
                        help="Select primary sample kit (tr-909 or tr-808). Default: tr-909")
    # REMOVED --kit-dir, now determined by --kit 
    # parser.add_argument("--kit-dir", type=str, default=None, help="Path to the sample kit directory for WAV rendering")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity level (-v, -vv)")

    # --- NEW: Dynamics Arguments ---
    dynamics_group = parser.add_argument_group('Dynamics Options')
    dynamics_group.add_argument("--panning", type=float, default=0.0, 
                              help="Maximum random stereo panning amount (0.0=center to 1.0=full left/right). Default: 0.0")
    dynamics_group.add_argument("--velocity-humanize", type=float, default=0.0, 
                              help="Amount of velocity randomization/humanization (0.0=none to 1.0=high). Default: 0.0")
    # --- End Dynamics --- 

    # --- NEW: Deep Percussion Evolution Arguments ---
    deep_perc_group = parser.add_argument_group('Deep Percussion Evolution')
    deep_perc_group.add_argument("--deep-perc-instruments", type=str, default="low_tom,low_conga,low_bongo",
                               help="Comma-separated list of instruments to apply evolution (e.g., 'low_tom,low_conga'). Default: low_tom,low_conga,low_bongo")
    deep_perc_group.add_argument("--deep-perc-prob-start", type=float, default=0.1,
                               help="Starting probability multiplier for deep percussion (0.0-1.0). Default: 0.1")
    deep_perc_group.add_argument("--deep-perc-prob-end", type=float, default=0.5,
                               help="Ending probability multiplier for deep percussion (0.0-1.0). Default: 0.5")
    deep_perc_group.add_argument("--deep-perc-vel-start", type=int, default=70,
                               help="Starting base velocity for deep percussion (1-127). Default: 70")
    deep_perc_group.add_argument("--deep-perc-vel-end", type=int, default=110,
                               help="Ending base velocity for deep percussion (1-127). Default: 110")
    deep_perc_group.add_argument("--deep-perc-timing-ms-start", type=float, default=15.0,
                               help="Starting timing randomness (ms) for deep percussion. Default: 15.0")
    deep_perc_group.add_argument("--deep-perc-timing-ms-end", type=float, default=3.0,
                               help="Ending timing randomness (ms) for deep percussion. Default: 3.0")
    # --- End Deep Percussion ---

    # --- NEW: Predictability Argument ---
    parser.add_argument("--predictability", type=float, default=0.5,
                        help="Controls groove predictability (0.0=max variation, 1.0=min variation). Default: 0.5")
    # --- End Predictability ---

    # --- NEW: General Evolution Arguments ---
    gen_evo_group = parser.add_argument_group('General Evolution (Fade In/Out)')
    gen_evo_group.add_argument("--evolve-instruments", type=str, default=None,
                               help="Comma-separated list of instruments for general evolution (e.g., 'open_hat,clap'). Default: None")
    gen_evo_group.add_argument("--evolve-param", type=str, default="probability", choices=["probability", "velocity"],
                               help="Parameter to evolve for general instruments ('probability' or 'velocity'). Default: probability")
    gen_evo_group.add_argument("--evolve-start", type=float, default=0.0,
                               help="Starting multiplier for the evolved parameter (0.0 to 1.0+). Default: 0.0")
    gen_evo_group.add_argument("--evolve-end", type=float, default=1.0,
                               help="Ending multiplier for the evolved parameter (0.0 to 1.0+). Default: 1.0")
    # --- End General Evolution ---

    # --- NEW: Adherence Argument ---
    parser.add_argument("--adherence", type=float, default=0.0, 
                        help="How strongly to enforce core patterns (0.0=probabilistic, 1.0=pattern-driven). Default: 0.0")
    # --- End Adherence ---

    args = parser.parse_args()

    # Clamp predictability and other values
    args.adherence = max(0.0, min(1.0, args.adherence)) # Clamp adherence
    args.predictability = max(0.0, min(1.0, args.predictability))
    args.panning = max(0.0, min(1.0, args.panning))
    args.velocity_humanize = max(0.0, min(1.0, args.velocity_humanize))

    # --- Determine Style Preset FIRST --- 
    selected_style = args.style
    if selected_style not in STYLE_PRESETS: 
        print(f"[WARNING] Style '{selected_style}' not found. Using 'dusty'.")
        selected_style = "dusty"
    style_preset = STYLE_PRESETS[selected_style]
    default_preset_swing = style_preset.get("swing_amount", 0.0)
    
    # --- Determine Default Included Drums (Used if structure specifies 'default' or no structure) ---
    default_included_drums = []
    # Check if any specific include/exclude flags were used (by comparing to defaults)
    user_overrode_defaults = (
        args.include_kick != True or 
        args.include_snare != True or 
        args.include_hat != True or 
        args.include_cymbal != False 
    )
    
    if user_overrode_defaults:
        if args.verbose >=1: print(f"[INFO] User overrides detected. Using specified include/exclude flags.")
        if args.include_kick: default_included_drums.append("kick")
        if args.include_snare: default_included_drums.append("snare")
        if args.include_hat: default_included_drums.extend(["closed_hat", "open_hat"])
        if args.include_cymbal: default_included_drums.extend(["crash_cymbal", "ride_cymbal"])
        # Add clap only if defined in style AND snare is included (as proxy, needs refinement)
        if "clap" in style_preset and "clap" not in default_included_drums and args.include_snare: 
            default_included_drums.append("clap")
    else:
        # If no specific flags changed from default, include ALL instruments defined in the style preset
        default_included_drums = [drum for drum in GM_DRUM_NOTES if drum in style_preset]
        if args.verbose >=1: print(f"[INFO] No specific include/exclude flags used. Including all instruments from style '{selected_style}': [{', '.join(default_included_drums)}]")

    # Always add clap if it's in the preset and wasn't added yet (covers cases like 'all')
    if "clap" in style_preset and "clap" not in default_included_drums:
        default_included_drums.append("clap")
        if args.verbose >=1: print(f"[INFO] Adding clap based on style preset: {selected_style}")

    if not default_included_drums: default_included_drums.append("kick")

    # --- Parse Structure or Use Defaults ---
    structure_data = []
    total_beats = 0
    if args.structure:
        try:
            sections = args.structure.split('/')
            for i, section_str in enumerate(sections):
                main_parts_str = section_str
                options_str = ""
                if ';' in section_str:
                    main_parts_str, options_str = section_str.split(';', 1)
                parts = main_parts_str.split('*', 2)
                if len(parts) != 3: raise ValueError(f"Section {i+1} ('{main_parts_str}') format incorrect. Use LENGTH*DENSITY*INSTRUMENTS[;options].")
                sec_len_str, sec_density, instruments_and_options = parts
                sec_len = int(sec_len_str)
                sec_density = sec_density.lower()
                if sec_density not in ["low", "medium", "high"]: raise ValueError(f"Invalid density '{sec_density}'.")
                sec_instruments_str = instruments_and_options
                options_str = ""
                if ';' in instruments_and_options:
                    sec_instruments_str, options_str = instruments_and_options.split(';', 1)
                sec_instruments_str = sec_instruments_str.lower().strip()
                sec_instruments = []
                if sec_instruments_str == 'default': sec_instruments = default_included_drums
                elif sec_instruments_str == 'all': sec_instruments = [drum for drum in GM_DRUM_NOTES if drum in style_preset]
                else: sec_instruments = [instr.strip() for instr in sec_instruments_str.split('+') if instr.strip() in GM_DRUM_NOTES]
                if not sec_instruments: raise ValueError(f"No valid instruments found for '{parts[2]}'")
                # Parse options (swing, variations, lead)
                sec_swing = default_preset_swing 
                sec_apply_variations = not style_preset.get("use_strict_pattern", False) 
                sec_lead_instrument = None # Default: no specific lead
                
                if options_str:
                    options = dict(item.split('=') for item in options_str.split(';') if '=' in item)
                    if 'swing' in options: 
                        try: sec_swing = float(options['swing']);
                        except ValueError: raise ValueError("Invalid swing value.")
                    if 'variations' in options:
                        if options['variations'].lower() == 'false': sec_apply_variations = False
                        elif options['variations'].lower() == 'true': sec_apply_variations = True
                        else: raise ValueError("Invalid variations value.")
                    if 'lead' in options:
                        lead_instr = options['lead'].lower().strip()
                        if lead_instr in GM_DRUM_NOTES: # Check if it's a valid instrument
                             sec_lead_instrument = lead_instr
                        else:
                             print(f"[WARNING] Invalid lead instrument '{lead_instr}' in section {i+1}. Ignoring.")

                structure_data.append({
                    "length": sec_len, "density": sec_density, "instruments": sec_instruments, 
                    "swing": sec_swing, "apply_variations": sec_apply_variations,
                    "lead_instrument": sec_lead_instrument # Add lead instrument
                })
                total_beats += sec_len
        except Exception as e:
            print(f"[ERROR] Invalid --structure format: {e}. Problem near: '{section_str}'")
            return
    else:
        # Fallback if no structure provided 
        # ... (fallback logic needs to add lead_instrument=None)
        structure_data.append({
            "length": args.length, 
            "density": args.density, 
            "instruments": default_included_drums,
            "swing": default_preset_swing, 
            "apply_variations": not style_preset.get("use_strict_pattern", False),
            "lead_instrument": None # Add default lead for fallback
        })
        total_beats = args.length
    
    # --- Create Output Directory --- (Moved after parsing)
    if not os.path.exists(args.output_dir):
        try: os.makedirs(args.output_dir); print(f"[INFO] Created output directory: {args.output_dir}")
        except OSError as e: print(f"[ERROR] Could not create output directory '{args.output_dir}': {e}"); return

    # --- Generate Groove --- 
    midi_data = generate_multi_drum_groove(
        style=selected_style, 
        bpm=args.bpm, 
        structure=structure_data, 
        default_included_drums=default_included_drums, 
        verbosity=args.verbose,
        # Add deep percussion evolution parameters
        deep_perc_instruments=args.deep_perc_instruments.split(','),
        deep_perc_prob_start=args.deep_perc_prob_start,
        deep_perc_prob_end=args.deep_perc_prob_end,
        deep_perc_vel_start=args.deep_perc_vel_start,
        deep_perc_vel_end=args.deep_perc_vel_end,
        deep_perc_timing_ms_start=args.deep_perc_timing_ms_start,
        deep_perc_timing_ms_end=args.deep_perc_timing_ms_end,
        # Pass predictability
        predictability=args.predictability,
        # Pass general evolution parameters
        evolve_instruments=args.evolve_instruments.split(',') if args.evolve_instruments else None,
        evolve_param=args.evolve_param,
        evolve_start=args.evolve_start,
        evolve_end=args.evolve_end,
        # Pass adherence
        adherence=args.adherence
    )

    # --- Output Handling --- 
    if not midi_data: print("[ERROR] MIDI generation failed."); return
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # --- Determine Selected Kit and Fallback Kit Paths ---
    selected_kit_name = args.kit.lower()
    primary_kit_path = None
    fallback_kit_path = None
    available_kits = {
        "tr-909": "sounds/Roland TR-909",
        "tr-808": "sounds/Roland TR-808"
    }

    if selected_kit_name in available_kits:
        primary_kit_path = available_kits[selected_kit_name]
        if selected_kit_name == "tr-909" and "tr-808" in available_kits:
            fallback_kit_path = available_kits["tr-808"]
        elif selected_kit_name == "tr-808" and "tr-909" in available_kits:
            fallback_kit_path = available_kits["tr-909"]
    else:
        # Should not happen due to choices in argparse, but good practice
        print(f"[ERROR] Invalid kit selected: {args.kit}. Using default tr-909.")
        primary_kit_path = available_kits["tr-909"]
        if "tr-808" in available_kits:
             fallback_kit_path = available_kits["tr-808"]

    if args.verbose >= 1:
        print(f"[INFO] Primary Kit: {primary_kit_path}")
        if fallback_kit_path:
            print(f"[INFO] Fallback Kit: {fallback_kit_path}")

    if not args.filename:
        safe_style = "".join(c if c.isalnum() else "_" for c in args.style)
        struct_info = f"struct{len(structure_data)}sec" if args.structure else f"{total_beats}beats"
        # Add kit name to default filename
        base_filename = f"multi_{safe_style}_{args.bpm}bpm_{args.density}_{struct_info}_{selected_kit_name}_{timestamp}"
    else:
        base_filename = os.path.splitext(args.filename)[0]
    
    midi_output_path = os.path.join(args.output_dir, f"{base_filename}.mid")
    wav_output_path = os.path.join(args.output_dir, f"{base_filename}.wav")
    midi_saved = False
    
    try:
        midi_data.save(midi_output_path)
        print(f"💾 MIDI file saved to: {midi_output_path}")
        midi_saved = True
    except Exception as e: 
        print(f"[ERROR] Failed to save MIDI file: {e}"); 
        # If MIDI save fails, don't attempt WAV render
        if args.format in ["wav", "both"]: return

    # --- WAV Rendering with Kit Selection and Fallback --- 
    if args.format in ["wav", "both"] and midi_saved:
        primary_sound_groups = None
        fallback_sound_groups = None

        if primary_kit_path and os.path.isdir(primary_kit_path):
            primary_sound_groups = group_samples_by_type(primary_kit_path, verbosity=args.verbose)
        else:
            print(f"[ERROR] Primary kit directory not found or invalid: {primary_kit_path}")
            return

        if fallback_kit_path and os.path.isdir(fallback_kit_path):
            fallback_sound_groups = group_samples_by_type(fallback_kit_path, verbosity=args.verbose)
        elif fallback_kit_path:
             if args.verbose >= 1: print(f"[INFO] Fallback kit directory not found: {fallback_kit_path}")

        if primary_sound_groups: # Only proceed if primary kit loaded
            render_success = render_midi_with_sample_groups(
                midi_file_path=midi_output_path, 
                output_wav_path=wav_output_path,
                primary_sound_groups=primary_sound_groups, 
                fallback_sound_groups=fallback_sound_groups, # Pass fallback groups
                verbosity=args.verbose,
                panning_amount=args.panning, 
                humanize_amount=args.velocity_humanize,
                style_preset=style_preset # Pass the correct preset here
            )
            if render_success: 
                print(f"🔊 WAV file saved to: {wav_output_path}")
            else:
                print(f"[ERROR] WAV rendering failed.")
        else:
             print(f"[ERROR] Failed to load primary sound groups. Cannot render WAV.")

    elif args.format in ["wav", "both"] and not midi_saved:
         print(f"[ERROR] MIDI file failed to save. Cannot render WAV.")

if __name__ == "__main__":
    main() 
