import argparse
import os
import random
import time
from mido import Message, MidiFile, MidiTrack, MetaMessage
import numpy as np
# Import the rendering function
from midi_to_wav import render_midi_with_sample_groups, group_samples_by_type

# --- Style Presets ---
# Define characteristics for different groove styles
STYLE_PRESETS = {
    "glitch": {
        "description": "Stuttering, broken rhythms with unexpected timing.",
        "note_probability": (0.1, 0.4), # Range based on density
        "velocity_range": (80, 110),
        "timing_randomness_ms": (5, 50), # How much notes can deviate (milliseconds)
        "double_hit_probability": 0.15,
        "skip_hit_probability": 0.10,
        "swing_amount": 0.0, # No swing for glitch
        "ghost_note_probability": 0.05,
        "ghost_note_velocity_factor": 0.3,
    },
    "dusty": {
        "description": "Lo-fi hip-hop style groove with swing and velocity variation.",
        "note_probability": (0.3, 0.7),
        "velocity_range": (60, 95),
        "timing_randomness_ms": (2, 20),
        "double_hit_probability": 0.05,
        "skip_hit_probability": 0.05,
        "swing_amount": 0.55, # More pronounced swing
        "ghost_note_probability": 0.1,
        "ghost_note_velocity_factor": 0.4,
    },
    # Add more styles here: "idm", "lofi", "minimal", "techno", etc.
    "minimal": {
        "description": "Sparse, clean rhythm.",
        "note_probability": (0.05, 0.2),
        "velocity_range": (90, 105),
        "timing_randomness_ms": (1, 10),
        "double_hit_probability": 0.01,
        "skip_hit_probability": 0.02,
        "swing_amount": 0.0,
        "ghost_note_probability": 0.0,
        "ghost_note_velocity_factor": 0.0,
    }
}

# --- Core Generation Logic ---

def generate_groove(style="glitch", bpm=120, density="medium", length_beats=16, midi_note=36, verbosity=1):
    """Generates a MIDI groove based on style, BPM, and density."""

    if verbosity >= 1:
        print(f"🥁 Generating groove: Style='{style}', BPM={bpm}, Density='{density}', Length={length_beats} beats")

    # --- Select Preset ---
    if style not in STYLE_PRESETS:
        print(f"[WARNING] Unknown style '{style}'. Defaulting to 'minimal'.")
        style = "minimal"
    preset = STYLE_PRESETS[style]
    if verbosity >= 1:
        print(f"[INFO] Using preset: {style} - {preset['description']}")

    # --- Map Density to Parameters ---
    density_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
    density_factor = density_map.get(density.lower(), 0.5)

    note_prob_min, note_prob_max = preset["note_probability"]
    current_note_prob = note_prob_min + (note_prob_max - note_prob_min) * density_factor

    timing_dev_min, timing_dev_max = preset["timing_randomness_ms"]
    current_timing_deviation_ms = timing_dev_min + (timing_dev_max - timing_dev_min) * density_factor

    vel_min, vel_max = preset["velocity_range"]

    # --- MIDI Setup ---
    TPB = 480 # Ticks Per Beat (standard)
    mid = MidiFile(ticks_per_beat=TPB)
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage('set_tempo', tempo=bpm_to_tempo(bpm), time=0))
    track.append(MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))

    # --- Generate Notes ---
    ticks_per_16th = TPB // 4
    total_ticks = length_beats * TPB
    
    last_event_abs_tick = 0 # Absolute tick of the last MIDI event added
    scheduled_notes = [] # Store notes as (absolute_tick, velocity, type) - type='normal', 'ghost', 'double'

    # 1. Schedule potential note positions (every 16th note)
    for beat in range(length_beats):
        for step in range(4): # 4 steps per beat (16th notes)
            abs_tick = (beat * 4 + step) * ticks_per_16th
            is_on_beat = (step == 0)
            is_off_beat_16th = (step % 2 != 0) # Is it the 2nd or 4th 16th note

            place_note = random.random() < current_note_prob
            is_ghost = False

            # Skip Hit?
            if place_note and random.random() < preset["skip_hit_probability"]:
                place_note = False
                if verbosity >= 2:
                     print(f"[DEBUG] Skipped hit at beat {beat+1}.{step+1}")

            # If not placing note, check for Ghost Note
            if not place_note and random.random() < preset["ghost_note_probability"]:
                place_note = True
                is_ghost = True
                if verbosity >= 2:
                    print(f"[DEBUG] Added ghost note at beat {beat+1}.{step+1}")

            if place_note:
                # Base Timing (incorporate swing)
                note_time = abs_tick
                if is_off_beat_16th and preset["swing_amount"] > 0:
                    swing_delay_ticks = int(ticks_per_16th * preset["swing_amount"] * (2/3))
                    note_time += swing_delay_ticks

                # Timing Randomness
                random_offset_ms = random.uniform(-current_timing_deviation_ms, current_timing_deviation_ms)
                random_offset_ticks = ms_to_ticks(random_offset_ms, bpm, TPB)
                note_time += random_offset_ticks
                note_time = max(0, note_time) # Ensure time doesn't go negative

                # Velocity
                velocity = random.randint(vel_min, vel_max)
                if is_ghost:
                    velocity = int(velocity * preset["ghost_note_velocity_factor"])
                    velocity = max(1, velocity) # Ensure velocity is at least 1

                scheduled_notes.append((note_time, velocity, 'ghost' if is_ghost else 'normal'))

                # Double Hit?
                if not is_ghost and random.random() < preset["double_hit_probability"]:
                    double_hit_offset_ticks = random.randint(ticks_per_16th // 4, ticks_per_16th // 2)
                    double_hit_time = note_time + double_hit_offset_ticks
                    double_hit_velocity = max(1, int(velocity * 0.7)) # Slightly lower velocity
                    if double_hit_time < total_ticks:
                        scheduled_notes.append((double_hit_time, double_hit_velocity, 'double'))
                        if verbosity >= 2:
                            print(f"[DEBUG] Added double hit at beat {beat+1}.{step+1}")

    # 2. Sort notes by time
    scheduled_notes.sort(key=lambda x: x[0])

    # 3. Add notes to MIDI track with correct delta times
    note_duration_ticks = ticks_per_16th // 2 # Fixed short duration for simplicity

    for note_time, velocity, note_type in scheduled_notes:
        note_time_rounded = int(round(note_time))
        note_off_time_rounded = note_time_rounded + note_duration_ticks

        # Ensure note_off doesn't exceed total length or overlap badly (simple check)
        if note_off_time_rounded >= total_ticks:
            note_off_time_rounded = total_ticks -1
            if note_time_rounded >= note_off_time_rounded:
                continue # Skip note if it starts too late

        # Calculate delta time for note_on
        delta_on = note_time_rounded - last_event_abs_tick
        if delta_on < 0: delta_on = 0 # Avoid negative delta time

        track.append(Message('note_on', note=midi_note, velocity=velocity, time=delta_on))
        last_event_abs_tick = note_time_rounded

        # Calculate delta time for note_off
        delta_off = note_off_time_rounded - last_event_abs_tick
        if delta_off < 0: delta_off = 0 # Should ideally be note_duration_ticks

        track.append(Message('note_off', note=midi_note, velocity=64, time=delta_off))
        last_event_abs_tick = note_off_time_rounded


    # --- Add End of Track ---
    # Ensure the end_of_track message has the correct delta time to reach the end
    end_delta = total_ticks - last_event_abs_tick
    if end_delta < 0: end_delta = 0
    track.append(MetaMessage('end_of_track', time=end_delta))

    if verbosity >= 1:
        print("✅ Groove generated.")
    return mid

# --- Helper Functions ---

def bpm_to_tempo(bpm):
    """Converts BPM to microseconds per beat for MIDI tempo meta message."""
    return int(60_000_000 / bpm)

def ms_to_ticks(ms, bpm, ticks_per_beat):
    """Converts milliseconds to MIDI ticks."""
    seconds_per_beat = 60.0 / bpm
    ticks_per_second = ticks_per_beat / seconds_per_beat
    return int(round(ms / 1000.0 * ticks_per_second))

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Groove Suggestion Agent: Generate MIDI grooves.")
    parser.add_argument("--style", type=str, default="glitch", help=f"Groove style keyword (e.g., {', '.join(STYLE_PRESETS.keys())})")
    parser.add_argument("--bpm", type=int, default=120, help="Beats per minute (tempo)")
    parser.add_argument("--density", type=str, default="medium", choices=["low", "medium", "high"], help="Groove density (low, medium, high)")
    parser.add_argument("--length", type=int, default=16, help="Length of the groove in beats (e.g., 16 for 4 bars of 4/4)")
    parser.add_argument("--note", type=int, default=36, help="MIDI note number to use (e.g., 36 for Kick Drum in GM)")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the output file(s)")
    parser.add_argument("--filename", type=str, default="", help="Custom base filename (optional, overrides default naming, extensions .mid/.wav added automatically)")
    parser.add_argument("--format", type=str, default="mid", choices=["mid", "wav", "both"], help="Output format (mid, wav, or both)")
    parser.add_argument("--soundfont", type=str, default=None, help=f"Path to the sample kit directory (e.g., sounds/Roland_TR-909/) for WAV rendering")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity level (-v, -vv)")

    args = parser.parse_args()

    # --- Create Output Directory ---
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            if args.verbose >= 1:
                print(f"[INFO] Created output directory: {args.output_dir}")
        except OSError as e:
            print(f"[ERROR] Could not create output directory '{args.output_dir}': {e}")
            return

    # --- Generate Groove ---
    midi_data = generate_groove(
        style=args.style,
        bpm=args.bpm,
        density=args.density,
        length_beats=args.length,
        midi_note=args.note,
        verbosity=args.verbose
    )

    if not midi_data:
        print("[ERROR] MIDI generation failed.")
        return

    # --- Determine Base Filename ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if not args.filename:
        safe_style = "".join(c if c.isalnum() else "_" for c in args.style)
        base_filename = f"{safe_style}_{args.bpm}bpm_{args.density}_{timestamp}"
    else:
        # Remove extensions if user accidentally included them
        base_filename = os.path.splitext(args.filename)[0]

    midi_output_path = os.path.join(args.output_dir, f"{base_filename}.mid")
    wav_output_path = os.path.join(args.output_dir, f"{base_filename}.wav")

    # --- Save MIDI File (if requested or needed for WAV) ---
    midi_saved = False
    if args.format in ["mid", "both"]:
        try:
            midi_data.save(midi_output_path)
            if args.verbose >= 0:
                print(f"💾 MIDI file saved to: {midi_output_path}")
            midi_saved = True
        except Exception as e:
            print(f"[ERROR] Failed to save MIDI file: {e}")
            # If we wanted WAV, we might need to stop here if MIDI saving fails
            if args.format == "wav":
                return 
    elif args.format == "wav":
         # Save MIDI temporarily even if only WAV is requested, as render function needs it
         try:
            midi_data.save(midi_output_path)
            midi_saved = True
         except Exception as e:
            print(f"[ERROR] Failed to save temporary MIDI file for WAV rendering: {e}")
            return # Cannot proceed to WAV rendering


    # --- Render WAV File (if requested and MIDI was saved) ---
    if args.format in ["wav", "both"] and midi_saved:
        # First group samples from the kit directory if soundfont argument is provided
        if args.soundfont and os.path.isdir(args.soundfont):
            sound_groups = group_samples_by_type(args.soundfont, verbosity=args.verbose)
            if sound_groups:
                render_success = render_midi_with_sample_groups(
                    midi_file_path=midi_output_path,
                    output_wav_path=wav_output_path,
                    sound_groups=sound_groups,
                    verbosity=args.verbose
                )
            else:
                print(f"[ERROR] Failed to group samples from directory: {args.soundfont}")
                render_success = False
        else:
            print("[ERROR] For WAV rendering, please provide a valid sample kit directory with --soundfont")
            render_success = False


if __name__ == "__main__":
    main() 
