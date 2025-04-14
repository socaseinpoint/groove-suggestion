import os
import sys
import argparse
import random
import re
from collections import defaultdict
from mido import MidiFile
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import mido
import numpy as np # Import numpy

# --- Constants ---
SAMPLE_RATE = 44100
SUPPORTED_AUDIO_EXTENSIONS = [".wav", ".mp3", ".aiff", ".flac", ".ogg"]

# --- Add TR-909 Specific Pattern Matchers ---
# Based on the TR909SET.TXT documentation:
# b=bass drum (kick), s=snare, l=low tom, m=mid tom, h=high tom, 
# rim=rimshot, hand=handclap, hhc=closed hi-hat, hho=open hi-hat,
# csh=crash cymbal, ride=ride cymbal
TR909_PATTERNS = [
    ('kick', re.compile(r'^b', re.IGNORECASE)),       # Bass drum samples start with 'b'
    ('snare', re.compile(r'^s', re.IGNORECASE)),      # Snare samples start with 's'
    ('tom', re.compile(r'^[lmh]t', re.IGNORECASE)),   # Toms start with lt, mt, or ht
    ('closed_hat', re.compile(r'^hhc', re.IGNORECASE)), # Closed hi-hat
    ('open_hat', re.compile(r'^hho', re.IGNORECASE)),   # Open hi-hat
    ('crash_cymbal', re.compile(r'^csh', re.IGNORECASE)), # Crash cymbal
    ('ride_cymbal', re.compile(r'^ride', re.IGNORECASE)), # Ride cymbal
    ('clap', re.compile(r'^hand', re.IGNORECASE)),      # Handclap
    ('rim', re.compile(r'^rim', re.IGNORECASE)),       # Rimshot
    # Add TR-909 patterns for BT*, ST*, etc. format (match by first letter)
    ('kick', re.compile(r'^bt', re.IGNORECASE)),     # Bass drum in TR-909 format
    ('snare', re.compile(r'^st', re.IGNORECASE)),    # Snare in TR-909 format
]

# --- MIDI Note to Sound Type Mapping (General MIDI Drum Map - partial) ---
GM_DRUM_NOTE_TO_TYPE = {
    # Kicks
    35: 'kick', 36: 'kick',
    # Snares / Rim
    37: 'rim', 38: 'snare', 40: 'snare', # Using 37 for Rim
    # Toms
    41: 'low_tom', 43: 'low_tom', 45: 'low_tom', # Grouping floor/low
    47: 'mid_tom', 48: 'mid_tom', # Grouping mid
    50: 'high_tom', # Grouping high
    # Hi-Hats
    42: 'closed_hat', 44: 'pedal_hat', 46: 'open_hat',
    # Cymbals
    49: 'crash_cymbal', 57: 'crash_cymbal', # Grouping crashes
    51: 'ride_cymbal', 59: 'ride_cymbal', # Grouping rides
    53: 'ride_bell',
    55: 'splash_cymbal',
    # Percussion
    39: 'clap',
    56: 'cowbell',
    75: 'claves',
    # ... (keep existing percussion like tambourine, etc. if samples exist)
    54: 'tambourine', 58: 'vibraslap', 60: 'hi_bongo', 61: 'low_bongo',
    62: 'mute_hi_conga', 63: 'open_hi_conga', 64: 'low_conga',
    65: 'high_timbale', 66: 'low_timbale', 67: 'high_agogo', 68: 'low_agogo',
    69: 'cabasa', 70: 'maracas', 71: 'short_whistle', 72: 'long_whistle',
    73: 'short_guiro', 74: 'long_guiro', 76: 'hi_wood_block', 77: 'low_wood_block',
    78: 'mute_cuica', 79: 'open_cuica', 80: 'mute_triangle', 81: 'open_triangle'
}

# --- Sample Grouping Logic ---
def group_samples_by_type(kit_dir, verbosity=1):
    """Scans a directory (and its subdirectories for specific kits like TR-808), \
       groups audio samples by type based on filename or directory conventions.
    """
    if not os.path.isdir(kit_dir):
        print(f"[ERROR] Sample kit directory not found: {kit_dir}")
        return None

    sound_groups = defaultdict(list)
    found_samples = 0

    # Regex patterns to identify common drum sample names
    # Order matters: more specific patterns first
    type_patterns = [
        # Hats first for specificity
        ('closed_hat', re.compile(r'(hat|hh|hihat).*(cl|close|c)', re.IGNORECASE)),
        ('open_hat', re.compile(r'(hat|hh|hihat).*(op|open|o)', re.IGNORECASE)),
        ('pedal_hat', re.compile(r'(hat|hh|hihat).*(pedal|foot|p|f)', re.IGNORECASE)),
        # Common Drums
        ('kick', re.compile(r'(kick|kik|bd|bass)', re.IGNORECASE)),
        ('snare', re.compile(r'(snare|snr|sd)', re.IGNORECASE)),
        ('clap', re.compile(r'clap', re.IGNORECASE)),
        # Toms (try to differentiate)
        ('high_tom', re.compile(r'(tom|t).*(h|hi|high)', re.IGNORECASE)),
        ('mid_tom', re.compile(r'(tom|t).*(m|mid)', re.IGNORECASE)),
        ('low_tom', re.compile(r'(tom|t).*(l|lo|low|f|floor)', re.IGNORECASE)),
        ('tom', re.compile(r'tom', re.IGNORECASE)), # Fallback for generic toms
        # Cymbals
        ('ride_cymbal', re.compile(r'(ride|rd)', re.IGNORECASE)),
        ('crash_cymbal', re.compile(r'(crash|cr)', re.IGNORECASE)),
        # --- NEW PERCUSSION --- 
        ('rim', re.compile(r'rim', re.IGNORECASE)),
        ('cowbell', re.compile(r'cow', re.IGNORECASE)),
        ('claves', re.compile(r'clave', re.IGNORECASE)),
        # --- END NEW --- 
        ('perc', re.compile(r'(perc|conga|bongo|agogo|tamb|shaker|cabasa|maracas|guiro|wood|cuica|triang|whistle|bell|splash|chinese)', re.IGNORECASE)), # General percussion catch-all
        # TR-909 specific (filename based)
        *TR909_PATTERNS,
        # TR-808 specific (directory based - map folder name to type)
        ('kick', re.compile(r'^BD$', re.IGNORECASE)),
        ('snare', re.compile(r'^SD$', re.IGNORECASE)),
        ('low_tom', re.compile(r'^LT$', re.IGNORECASE)),
        ('mid_tom', re.compile(r'^MT$', re.IGNORECASE)),
        ('high_tom', re.compile(r'^HT$', re.IGNORECASE)),
        ('closed_hat', re.compile(r'^CH$', re.IGNORECASE)),
        ('open_hat', re.compile(r'^OH$', re.IGNORECASE)),
        ('crash_cymbal', re.compile(r'^CY$', re.IGNORECASE)),
        ('clap', re.compile(r'^CP$', re.IGNORECASE)), # CP likely clap
        ('claves', re.compile(r'^CL$', re.IGNORECASE)), # CL likely claves
        ('rim', re.compile(r'^RS$', re.IGNORECASE)), # RS likely rimshot
        ('cowbell', re.compile(r'^CB$', re.IGNORECASE)),
        ('maracas', re.compile(r'^MA$', re.IGNORECASE)),
        ('low_conga', re.compile(r'^LC$', re.IGNORECASE)),
        ('mid_conga', re.compile(r'^MC$', re.IGNORECASE)),
        ('high_conga', re.compile(r'^HC$', re.IGNORECASE)),
    ]

    # Use os.walk to handle potential subdirectories (for TR-808 structure)
    for root, dirs, files in os.walk(kit_dir):
        # Determine type from directory name if applicable (e.g., for TR-808)
        dir_name = os.path.basename(root)
        dir_matched_type = None
        if root != kit_dir: # Only check subdirectories
            for type_name, pattern in type_patterns:
                if pattern.fullmatch(dir_name): # Use fullmatch for directory names
                    dir_matched_type = type_name
                    break

        for filename in files:
            base_name, ext = os.path.splitext(filename)
            if ext.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                full_path = os.path.join(root, filename)
                matched_type = dir_matched_type # Prioritize type from directory

                if not matched_type: # Fallback to filename matching if no dir match
                    for type_name, pattern in type_patterns:
                        if pattern.search(base_name):
                            matched_type = type_name
                            break
                
                if matched_type:
                    sound_groups[matched_type].append(full_path)
                    found_samples += 1
                    if verbosity >= 2:
                        source_info = f" (from dir '{dir_name}')" if dir_matched_type else ""
                        print(f"[DEBUG] Grouped '{filename}'{source_info} as '{matched_type}'")
                elif verbosity >= 2:
                    print(f"[DEBUG] Could not automatically group sample: {filename} in {dir_name}")

    if found_samples == 0:
        print(f"[WARNING] No supported audio samples found or grouped in directory: {kit_dir}")
        return None

    if verbosity >= 1:
        print(f"[INFO] Grouped {found_samples} samples from '{os.path.basename(kit_dir)}' into {len(sound_groups)} types:")
        for type_name, samples in sound_groups.items():
            print(f"         - {type_name}: {len(samples)} samples")
           
    return sound_groups

def get_midi_info(midi_file_path):
    """Reads MIDI file to get tempo and duration."""
    try:
        mid = MidiFile(midi_file_path)
    except Exception as e:
        print(f"[ERROR] Could not read MIDI file {midi_file_path}: {e}")
        return None, None, 0

    ticks_per_beat = mid.ticks_per_beat or 480 # Default if not specified
    tempo = 500000 # Default tempo (120 bpm) if not found
    total_ticks = 0
    
    # First pass: find the tempo
    for track in mid.tracks:
        for msg in track:
            if msg.is_meta and msg.type == 'set_tempo':
                tempo = msg.tempo
                break # Use first tempo found
        # If tempo was found in this track, no need to check others
        if tempo != 500000:
            break
    
    # Second pass: calculate total duration based on the maximum duration of all tracks
    max_track_ticks = 0
    for track in mid.tracks:
        track_ticks = 0
        for msg in track:
            track_ticks += msg.time
        max_track_ticks = max(max_track_ticks, track_ticks)
    
    seconds_per_beat = tempo / 1_000_000.0
    seconds_per_tick = seconds_per_beat / ticks_per_beat
    total_duration_seconds = max_track_ticks * seconds_per_tick
    
    return tempo, ticks_per_beat, total_duration_seconds

def render_midi_with_sample_groups(midi_file_path, output_wav_path, 
                                   primary_sound_groups, 
                                   fallback_sound_groups=None, # Add fallback groups
                                   note_to_type_map=GM_DRUM_NOTE_TO_TYPE, 
                                   sample_rate=SAMPLE_RATE, verbosity=1,
                                   # --- NEW Dynamics Params ---
                                   panning_amount=0.0, 
                                   humanize_amount=0.0,
                                   style_preset={}):
    if verbosity >= 1: print(f"🥁 Rendering {os.path.basename(midi_file_path)} to WAV using Explicit NumPy Mixing (Panning={panning_amount:.2f}, Humanize={humanize_amount:.2f})...")
    if not primary_sound_groups: print("[ERROR] Invalid or empty primary sound groups provided."); return False

    # --- Stage 1: Collect Note Info (start_sample, note, velocity) ---
    notes_to_process = []
    max_abs_ticks = 0
    try:
        mid = MidiFile(midi_file_path)
        ticks_per_beat = mid.ticks_per_beat or 480
        tempo, _, _ = get_midi_info(midi_file_path)
        seconds_per_tick = (tempo / 1_000_000.0) / ticks_per_beat
        if verbosity >= 1: print(f"[INFO] Tempo={int(60_000_000 / tempo)} BPM, TPB={ticks_per_beat}, Secs/Tick={seconds_per_tick:.8f}")

        current_abs_ticks = 0 # Use a separate variable to track absolute time
        for msg in mido.merge_tracks(mid.tracks):
            # Process message time BEFORE calculating absolute time for the note
            current_abs_ticks += msg.time
            max_abs_ticks = max(max_abs_ticks, current_abs_ticks)

            if msg.type == 'note_on' and msg.velocity > 0:
                # Calculate start time based on the ABSOLUTE ticks accumulated so far
                start_sample = int(round(current_abs_ticks * seconds_per_tick * sample_rate))
                start_time_ms = current_abs_ticks * seconds_per_tick * 1000.0
                notes_to_process.append((start_sample, msg.note, msg.velocity))
                if verbosity >= 3:
                     # Print the correct absolute ticks used for this note's timing
                     print(f"[TIMING DEBUG] Note {msg.note} Vel {msg.velocity} | Delta Ticks: {msg.time}, **Used Abs Ticks**: {current_abs_ticks}, Start MS: {start_time_ms:.2f}, Start Sample: {start_sample}")
                
    except Exception as e:
        print(f"[ERROR] Failed during MIDI processing (Stage 1): {e}"); return False

    if not notes_to_process: print("[WARNING] No notes found in MIDI file."); return False
    if verbosity >= 1: print(f"[INFO] Stage 1 complete: Found {len(notes_to_process)} notes.")

    # --- Stage 2: Prepare Audio Data Cache {(note): numpy_array} ---
    # --- RE-INTRODUCE SHORTENING for KICK/SNARE --- 
    audio_data_cache = {}
    loaded_samples_cache = {} 
    max_note_end_sample = 0
    notes_processed_stage2 = 0

    # Create a cache of unique notes needed to avoid redundant loading
    unique_notes_needed = set(note for _, note, _ in notes_to_process)

    for note in unique_notes_needed:
        cache_key = note 
        if cache_key in audio_data_cache: continue 

        sound_type = note_to_type_map.get(note)
        chosen_sample_path = None
        source_kit = "primary"

        # Try primary kit first
        if sound_type and sound_type in primary_sound_groups and primary_sound_groups[sound_type]:
            chosen_sample_path = random.choice(primary_sound_groups[sound_type])
        # Try fallback kit if primary fails and fallback exists
        elif fallback_sound_groups and sound_type and sound_type in fallback_sound_groups and fallback_sound_groups[sound_type]:
            chosen_sample_path = random.choice(fallback_sound_groups[sound_type])
            source_kit = "fallback"
            if verbosity >=1: print(f"[INFO] Note {note} (type '{sound_type}'): Using fallback sample from other kit.")
        else:
             if verbosity >= 1: print(f"[WARNING] No sample group found for note {note}, type '{sound_type}' in primary or fallback kits. Skipping prep.")
             continue

        try:
            # Load the chosen sample (might already be in cache)
            if chosen_sample_path in loaded_samples_cache:
                sample = loaded_samples_cache[chosen_sample_path]
            else:
                sample = AudioSegment.from_file(chosen_sample_path).set_frame_rate(sample_rate)
                loaded_samples_cache[chosen_sample_path] = sample
                if verbosity >= 2: print(f"[DEBUG] Loaded sample for note {note} from {source_kit} kit: {os.path.basename(chosen_sample_path)}")
        except Exception as e:
            print(f"[ERROR] Failed load sample for note {note} ({chosen_sample_path}): {e}. Skipping prep.")
            continue

        # --- Apply Shortening based on sound_type --- 
        processed_sample = sample # Start with original sample
        if sound_type == 'kick':
            max_length_ms = 150
            if len(processed_sample) > max_length_ms:
                 processed_sample = processed_sample[:max_length_ms].fade_out(30)
        elif sound_type == 'snare':
            max_length_ms = 250 
            if len(processed_sample) > max_length_ms:
                processed_sample = processed_sample[:max_length_ms].fade_out(40)
        # --- End Shortening --- 
        
        # Convert final processed sample to normalized float32 NumPy array (mono)
        try:
            processed_sample = processed_sample.set_channels(1) # Ensure mono
            audio_array = np.array(processed_sample.get_array_of_samples()).astype(np.float32)
            audio_array /= (2**(processed_sample.sample_width * 8 - 1)) # Normalize
            audio_data_cache[cache_key] = audio_array
            notes_processed_stage2 += 1
        except Exception as e:
             print(f"[ERROR] Failed converting sample for note {note} to NumPy: {e}. Skipping.")
             continue

    if not audio_data_cache: print("[ERROR] Failed to prepare any audio data."); return False
    if verbosity >= 1: print(f"[INFO] Stage 2 complete: Prepared audio data for {notes_processed_stage2} unique notes (with shortening).")

    # --- Stage 3: Mix into Final NumPy Array --- 
    # Determine required length, considering effect tails
    max_req_len = 0
    max_fx_tail_samples = 0 
    for start_sample, note, velocity in notes_to_process:
        cache_key = note 
        if cache_key in audio_data_cache:
             audio_array = audio_data_cache[cache_key]
             velocity_scale = velocity / 127.0 # Use original velocity for length calc
             scaled_audio_array = audio_array * velocity_scale # Use original velocity for length calc
             note_len_samples = len(scaled_audio_array)
             max_req_len = max(max_req_len, start_sample + note_len_samples)
             
             sound_type = note_to_type_map.get(note)
             if sound_type in style_preset:
                 drum_preset = style_preset[sound_type]
                 delay_params = drum_preset.get("delay_params")
                 reverb_params = drum_preset.get("reverb_params")
                 
                 current_max_end = start_sample + note_len_samples
                 
                 # Estimate delay tail 
                 if delay_params and delay_params.get("mix", 0) > 0:
                     delay_ms = delay_params.get("time_ms", 100)
                     delay_samples = int(delay_ms / 1000.0 * sample_rate)
                     feedback = delay_params.get("feedback", 0.5)
                     if feedback < 0.95: # Rough estimate for tail end
                        num_taps = 4 # Estimate tail based on 4 taps
                        current_max_end = max(current_max_end, start_sample + note_len_samples + num_taps * delay_samples)
                 
                 # Estimate reverb tail
                 if reverb_params and reverb_params.get("mix", 0) > 0:
                     reverb_ms = reverb_params.get("time_ms", 200) # Use initial reverb time
                     reverb_samples = int(reverb_ms / 1000.0 * sample_rate)
                     # Estimate tail based on a few simulated taps (like delay)
                     num_reverb_taps = 4 # Simulate a few reflections
                     reverb_decay_factor = 0.6 # Assume reverb taps decay faster
                     est_reverb_tail = 0
                     current_reverb_delay = reverb_samples
                     for _ in range(num_reverb_taps):
                        est_reverb_tail += current_reverb_delay
                        current_reverb_delay = int(current_reverb_delay * reverb_decay_factor) 
                     current_max_end = max(current_max_end, start_sample + note_len_samples + est_reverb_tail) 
                 
                 max_fx_tail_samples = max(max_fx_tail_samples, current_max_end)
                     
    # Use the maximum end position found (dry note end or effect tail end)
    total_samples = max(max_req_len, max_fx_tail_samples) + int(sample_rate * 0.2) # Increase buffer slightly
    if verbosity >= 1: print(f"[INFO] Stage 3: Creating final STEREO audio canvas ({total_samples} samples)...")
    final_audio_data = np.zeros((total_samples, 2), dtype=np.float32) 

    # --- Mixing Loop --- 
    notes_mixed = 0
    for start_sample, note, velocity in notes_to_process:
        cache_key = note 
        if cache_key in audio_data_cache:
            audio_array_mono = audio_data_cache[cache_key]
            final_velocity = velocity
            if humanize_amount > 0:
                max_vel_delta = int(humanize_amount * 64 * (velocity / 127.0))
                if max_vel_delta > 0:
                     vel_change = random.randint(-max_vel_delta, max_vel_delta)
                     final_velocity = max(1, min(127, velocity + vel_change))
            velocity_scale = final_velocity / 127.0
            scaled_audio_array = audio_array_mono * velocity_scale
            note_len_samples = len(scaled_audio_array)

            # Prepare Stereo Data (Panning)
            pan = random.uniform(-panning_amount, panning_amount) if panning_amount > 0 else 0.0
            pan_rad = (pan + 1.0) / 2.0 * (np.pi / 2.0)
            gain_left = np.cos(pan_rad)
            gain_right = np.sin(pan_rad)
            note_data_stereo = np.zeros((note_len_samples, 2), dtype=np.float32)
            note_data_stereo[:, 0] = scaled_audio_array * gain_left
            note_data_stereo[:, 1] = scaled_audio_array * gain_right
            
            # --- Apply Effects --- 
            sound_type = note_to_type_map.get(note)
            dry_mix = 1.0
            wet_signals_to_mix = [] 
            
            if sound_type in style_preset:
                drum_preset = style_preset[sound_type]
                delay_params = drum_preset.get("delay_params")
                reverb_params = drum_preset.get("reverb_params")
                
                # --- Simple Delay --- 
                if delay_params and delay_params.get("mix", 0) > 0:
                    delay_ms = delay_params.get("time_ms", 150)
                    feedback = delay_params.get("feedback", 0.4)
                    mix = delay_params.get("mix", 0.3)
                    delay_samples = int(delay_ms / 1000.0 * sample_rate)
                    dry_mix -= mix 
                    current_delay_signal = note_data_stereo * mix 
                    current_feedback = 1.0 
                    # Generate multiple delay taps
                    for i in range(4): # Simulate more taps
                         delay_start = start_sample + (i + 1) * delay_samples
                         if delay_start < total_samples:
                             wet_signals_to_mix.append((delay_start, current_delay_signal * current_feedback))
                             current_feedback *= feedback 
                             if current_feedback < 0.01: break # Stop if feedback is negligible
                         else: break 
                         
                # --- Improved Crude Reverb (Multi-tap echo) --- 
                if reverb_params and reverb_params.get("mix", 0) > 0:
                    reverb_initial_ms = reverb_params.get("time_ms", 300)
                    mix = reverb_params.get("mix", 0.2)
                    reverb_decay = reverb_params.get("decay", 0.6)
                    num_taps = 5 
                    dry_mix -= mix 
                    
                    if verbosity >= 3: print(f"[REVERB DEBUG] Applying Reverb Note {note} ({sound_type}): Time={reverb_initial_ms}ms Mix={mix:.2f} Decay={reverb_decay:.2f}")
                    
                    current_reverb_signal = note_data_stereo * mix 
                    current_delay_samples = 0
                    tap_delay_factor = 1.5 
                    tap_time_ms = reverb_initial_ms * 0.3 
                    
                    for i in range(num_taps):
                        current_delay_samples += int(tap_time_ms / 1000.0 * sample_rate)
                        reverb_start = start_sample + current_delay_samples
                        if reverb_start < total_samples:
                            tap_gain = reverb_decay**i
                            tap_signal = current_reverb_signal * tap_gain
                            if verbosity >= 3: print(f"  -> Reverb Tap {i+1}: StartSample={reverb_start}, Gain={tap_gain:.3f}")
                            wet_signals_to_mix.append((reverb_start, tap_signal))
                            tap_time_ms *= tap_delay_factor 
                        else: 
                            if verbosity >= 3: print(f"  -> Reverb Tap {i+1}: Skipped (start sample {reverb_start} > total {total_samples})")
                            break
            
            dry_mix = max(0.0, dry_mix)
            
            # --- Add Confirmation Debug ---
            if verbosity >= 3 and wet_signals_to_mix:
                print(f"[FX DEBUG] Note {note}: Generated {len(wet_signals_to_mix)} wet signal taps to mix.")
            elif verbosity >= 3:
                 print(f"[FX DEBUG] Note {note}: No wet signals generated.")
            # ---------------------------
            
            # --- Mix Dry Signal --- 
            end_sample_dry = start_sample + note_len_samples
            if end_sample_dry > final_audio_data.shape[0]:
                len_to_mix_dry = final_audio_data.shape[0] - start_sample
                if len_to_mix_dry > 0: final_audio_data[start_sample:final_audio_data.shape[0]] += note_data_stereo[:len_to_mix_dry] * dry_mix
            else:
                 final_audio_data[start_sample:end_sample_dry] += note_data_stereo * dry_mix
            for fx_start_sample, fx_data in wet_signals_to_mix:
                 fx_len = len(fx_data)
                 fx_end_sample = fx_start_sample + fx_len
                 if fx_end_sample > final_audio_data.shape[0]:
                     len_to_mix_fx = final_audio_data.shape[0] - fx_start_sample
                     if len_to_mix_fx > 0: final_audio_data[fx_start_sample:final_audio_data.shape[0]] += fx_data[:len_to_mix_fx]
                 else:
                     final_audio_data[fx_start_sample:fx_end_sample] += fx_data
                     
            notes_mixed += 1

    if verbosity >= 1: print(f"[INFO] Stage 3 complete: Mixed {notes_mixed} notes.")

    # --- Stage 4: Normalize and Export ---
    try:
        if verbosity >= 1: print(f"[INFO] Stage 4: Normalizing and converting final audio...")
        peak_amplitude = np.max(np.abs(final_audio_data)) # Find overall peak
        if peak_amplitude == 0: print("[WARNING] Output audio is silent.")
        elif peak_amplitude > 1.0:
            normalization_factor = 0.98 / peak_amplitude
            final_audio_data *= normalization_factor
            if verbosity >=1: print(f"[INFO] Audio normalized (peak was {peak_amplitude:.2f}, factor {normalization_factor:.2f})")
        
        int16_data = (final_audio_data * (2**15 - 1)).astype(np.int16)

        # Create STEREO AudioSegment 
        final_audio = AudioSegment(
            int16_data.tobytes(), 
            frame_rate=sample_rate,
            sample_width=2, 
            channels=2 # STEREO 
        )
        output_dir = os.path.dirname(output_wav_path)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        final_audio.export(output_wav_path, format="wav")
        if verbosity >= 1: print(f"✅ WAV file saved to: {output_wav_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed during NumPy conversion or export (Stage 4): {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render a MIDI file to WAV using randomly selected samples from a kit directory.")
    parser.add_argument("midi_file", help="Path to the input MIDI file.")
    parser.add_argument("-k", "--kit-dir", required=True, help="Path to the directory containing the sample kit (e.g., sounds/Roland_TR-909).")
    parser.add_argument("-o", "--output", default=None, help="Path for the output WAV file (defaults to input filename with .wav extension).")
    parser.add_argument("-r", "--rate", type=int, default=SAMPLE_RATE, help=f"Sample rate for the output WAV (default: {SAMPLE_RATE}).")
    parser.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity level (-vv for more details). Starts at level 1.")
    # --- NEW: Add Dynamics Arguments --- 
    parser.add_argument("--panning", type=float, default=0.0, help="Max random stereo panning (0.0 to 1.0).")
    parser.add_argument("--velocity-humanize", type=float, default=0.0, help="Velocity randomization amount (0.0 to 1.0).")
    # ---------------------------------

    args = parser.parse_args()

    # Clamp panning and humanization values
    args.panning = max(0.0, min(1.0, args.panning))
    args.velocity_humanize = max(0.0, min(1.0, args.velocity_humanize))
    
    # Determine output path
    output_wav_path = args.output
    if output_wav_path is None:
        base_name = os.path.splitext(args.midi_file)[0]
        output_wav_path = f"{base_name}.wav"

    # Check if input MIDI file exists
    if not os.path.exists(args.midi_file):
        print(f"[ERROR] Input MIDI file not found: {args.midi_file}")
        sys.exit(1)

    # Group samples from the kit directory
    sound_groups = group_samples_by_type(args.kit_dir, verbosity=args.verbose)
    if sound_groups is None:
        sys.exit(1) # Error message already printed by grouping function

    # Get style preset from groove agent file if possible (requires access/import - complex)
    # For simplicity, we pass None for style_preset here. Effects defined in presets won't apply unless passed.
    # A more robust solution would involve sharing preset data or passing it via another mechanism.
    style_preset_data = {} # Pass empty dict for now
    
    success = render_midi_with_sample_groups(
        midi_file_path=args.midi_file,
        output_wav_path=output_wav_path,
        primary_sound_groups=sound_groups, 
        note_to_type_map=GM_DRUM_NOTE_TO_TYPE,
        sample_rate=args.rate,
        verbosity=args.verbose,
        # Pass dynamics args to renderer
        panning_amount=args.panning,
        humanize_amount=args.velocity_humanize,
        style_preset=style_preset_data # Pass empty dict
    )

    if not success:
        sys.exit(1)
    else:
        sys.exit(0) 
