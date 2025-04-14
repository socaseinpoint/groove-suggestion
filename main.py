import os
import uuid
import tempfile
import traceback
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from functools import wraps
import glob

# Set verbosity level for debug information (0-3)
DEBUG_LEVEL = 3

def debug_print(level, message):
    """Print debug messages based on verbosity level"""
    if DEBUG_LEVEL >= level:
        print(f"[DEBUG {level}] {message}")

# Assuming midi_to_wav.py exists in the same directory
try:
    debug_print(1, "Attempting to import from midi_to_wav...")
    from midi_to_wav import render_midi_with_sample_groups, group_samples_by_type
    debug_print(2, "Successfully imported from midi_to_wav")
except ImportError as e:
    debug_print(1, f"WARNING: midi_to_wav.py import failed: {e}")
    
    # Define dummy functions if import fails to avoid runtime errors
    def render_midi_with_sample_groups(*args, **kwargs):
        debug_print(1, f"ERROR: midi_to_wav.render_midi_with_sample_groups is not available. Args: {args}, Kwargs: {kwargs}")
        return False
    
    def group_samples_by_type(*args, **kwargs):
        debug_print(1, f"ERROR: midi_to_wav.group_samples_by_type is not available. Args: {args}, Kwargs: {kwargs}")
        return None

# Import necessary functions/data from your groove generation script
try:
    debug_print(1, "Attempting to import from groove_agent_multi...")
    from groove_agent_multi import (
        generate_multi_drum_groove,
        STYLE_PRESETS,
        GM_DRUM_NOTES,
        # Import helper functions needed for parsing/setup if main() logic is moved here
        # bpm_to_tempo, ms_to_ticks, etc.
    )
    debug_print(2, "Successfully imported from groove_agent_multi")
    debug_print(3, f"Available styles: {list(STYLE_PRESETS.keys())}")
    debug_print(3, f"Available GM drum notes: {list(GM_DRUM_NOTES.keys())}")
except ImportError as e:
    debug_print(1, f"ERROR: groove_agent_multi.py import failed: {e}")
    # Define dummy generate function if import fails
    def generate_multi_drum_groove(*args, **kwargs):
        debug_print(1, f"ERROR: groove_agent_multi.generate_multi_drum_groove is not available. Args: {args}, Kwargs: {kwargs}")
        return None
    STYLE_PRESETS = {}
    GM_DRUM_NOTES = {}
    debug_print(1, "Initialized empty STYLE_PRESETS and GM_DRUM_NOTES")
except Exception as e:
    debug_print(1, f"ERROR during groove_agent_multi import: {e}")
    traceback.print_exc()
    STYLE_PRESETS = {}
    GM_DRUM_NOTES = {}
    debug_print(1, "Initialized empty STYLE_PRESETS and GM_DRUM_NOTES due to error")


app = FastAPI()

# --- Health Check Endpoint --- 
@app.get("/health")
async def health_check():
    """Simple health check endpoint for Render"""
    return {"status": "ok"}

# Define the expected input parameters using Pydantic
class GenerateParams(BaseModel):
    style: str = "dusty"
    bpm: int = 120
    structure: Optional[str] = None
    # --- Fallback/Default parameters if structure is not provided ---
    density: str = "medium"
    length: int = 16
    # --- Sound inclusion flags (for 'default' in structure or no structure) ---
    include_kick: bool = True
    include_snare: bool = True
    include_hat: bool = True
    include_cymbal: bool = False
    # --- Kit Selection ---
    kit: str = "tr-909" # Choices: "tr-909", "tr-808"
    # --- Dynamics ---
    panning: float = 0.0
    velocity_humanize: float = 0.0
    # --- Deep Percussion Evolution ---
    deep_perc_instruments: Optional[str] = "low_tom,low_conga,low_bongo"
    deep_perc_prob_start: float = 0.1
    deep_perc_prob_end: float = 0.5
    deep_perc_vel_start: int = 70
    deep_perc_vel_end: int = 110
    deep_perc_timing_ms_start: float = 15.0
    deep_perc_timing_ms_end: float = 3.0
    # --- Predictability ---
    predictability: float = 0.5
    # --- General Evolution ---
    evolve_instruments: Optional[str] = None
    evolve_param: str = "probability" # Choices: "probability", "velocity"
    evolve_start: float = 0.0
    evolve_end: float = 1.0
    # --- Adherence ---
    adherence: float = 0.0
    # --- Phrase Length ---
    phrase_length_bars: int = 4


def cleanup_file(file_path: Path):
    """Function to remove a file, used for background tasks."""
    try:
        os.remove(file_path)
        print(f"Cleaned up temporary file: {file_path}")
    except OSError as e:
        print(f"Error cleaning up file {file_path}: {e}")

@app.post("/generate")
async def generate_endpoint(params: GenerateParams, background_tasks: BackgroundTasks):
    """
    Generates a drum groove WAV file based on input parameters.
    """
    debug_print(1, f"Received generation request with params: {params.dict()}")

    try:
        # --- Parameter Parsing and Preparation (adapted from groove_agent_multi.main) ---
        try:
            selected_style = params.style
            debug_print(2, f"Selected style: {selected_style}, Available styles: {list(STYLE_PRESETS.keys())}")
            
            if selected_style not in STYLE_PRESETS:
                debug_print(1, f"[WARNING] Style '{selected_style}' not found. Using 'dusty'.")
                selected_style = "dusty"
                
                # Extra check if STYLE_PRESETS is empty
                if not STYLE_PRESETS:
                    raise HTTPException(
                        status_code=500, 
                        detail="STYLE_PRESETS dictionary is empty. This may indicate a problem with groove_agent_multi.py import."
                    )
            
            style_preset = STYLE_PRESETS.get(selected_style, {}) # Use .get for safety
            if not style_preset:
                 raise HTTPException(status_code=500, detail=f"Style preset '{selected_style}' is empty or invalid.")

            debug_print(3, f"Style preset: {style_preset}")
            default_preset_swing = style_preset.get("swing_amount", 0.0)
            debug_print(2, f"Default preset swing: {default_preset_swing}")
        except Exception as e:
            debug_print(1, f"ERROR in style/preset handling: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error in style/preset handling: {e}")

        try:
            # Determine default included drums based on request params
            default_included_drums = []
            if params.include_kick: default_included_drums.append("kick")
            if params.include_snare: default_included_drums.append("snare") # Includes rim/clap logic later?
            if params.include_hat: default_included_drums.extend(["closed_hat", "open_hat"])
            if params.include_cymbal: default_included_drums.extend(["crash_cymbal", "ride_cymbal"])
            # Add clap if defined in style AND snare is included (as proxy)
            if "clap" in style_preset and "clap" not in default_included_drums and params.include_snare:
                default_included_drums.append("clap")
            if not default_included_drums: default_included_drums.append("kick") # Ensure at least one drum

            debug_print(2, f"Default included drums (before additions): {default_included_drums}")
            
            # Add other defined percussion from the style preset to the defaults
            if not GM_DRUM_NOTES:
                debug_print(1, f"WARNING: GM_DRUM_NOTES is empty - no additional drums will be added")
            else:
                debug_print(3, f"Checking additional drums from GM_DRUM_NOTES: {list(GM_DRUM_NOTES.keys())}")
                for drum in GM_DRUM_NOTES:
                    if drum in style_preset and drum not in default_included_drums:
                        debug_print(2, f"Adding additional drum: {drum}")
                        default_included_drums.append(drum)

            debug_print(1, f"Final default included drums: {default_included_drums}")
        except Exception as e:
            debug_print(1, f"ERROR in drum selection: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error in drum selection: {e}")

        try:
            structure_data = []
            total_beats = 0
            if params.structure:
                debug_print(2, f"Parsing structure: {params.structure}")
                try:
                    sections = params.structure.split('/')
                    for i, section_str in enumerate(sections):
                        # Basic parsing logic - This needs careful adaptation from groove_agent_multi.main
                        main_parts_str = section_str
                        options_str = ""
                        if ';' in section_str:
                            main_parts_str, options_str = section_str.split(';', 1)

                        parts = main_parts_str.split('*')
                        debug_print(3, f"Section parts: {parts}")
                        if len(parts) != 3: raise ValueError(f"Section {i+1} format incorrect.")

                        sec_len = int(parts[0])
                        sec_density = parts[1].lower()
                        if sec_density not in ["low", "medium", "high"]: raise ValueError(f"Invalid density '{sec_density}'.")

                        sec_instruments_str = parts[2].lower().strip()
                        sec_instruments = []
                        if sec_instruments_str == 'default': sec_instruments = default_included_drums
                        elif sec_instruments_str == 'all': sec_instruments = [drum for drum in GM_DRUM_NOTES if drum in style_preset]
                        else: sec_instruments = [instr.strip() for instr in sec_instruments_str.split('+') if instr.strip() in GM_DRUM_NOTES and instr.strip() in style_preset] # Ensure valid AND in preset

                        if not sec_instruments: raise ValueError(f"No valid instruments in style '{selected_style}' found for section '{parts[2]}'")

                        # Parse options (swing, variations, lead) - needs error handling
                        sec_swing = default_preset_swing
                        sec_apply_variations = not style_preset.get("use_strict_pattern", False)
                        sec_lead_instrument = None
                        if options_str:
                            options = dict(item.split('=') for item in options_str.split(';') if '=' in item)
                            if 'swing' in options: sec_swing = float(options['swing'])
                            if 'variations' in options: sec_apply_variations = options['variations'].lower() == 'true'
                            if 'lead' in options:
                                lead_instr = options['lead'].lower().strip()
                                if lead_instr in sec_instruments: # Must be a valid instrument *for this section*
                                    sec_lead_instrument = lead_instr

                        structure_data.append({
                            "length": sec_len, "density": sec_density, "instruments": sec_instruments,
                            "swing": sec_swing, "apply_variations": sec_apply_variations,
                            "lead_instrument": sec_lead_instrument
                        })
                        total_beats += sec_len
                except Exception as e:
                    debug_print(1, f"ERROR in structure parsing: {e}")
                    traceback.print_exc()
                    raise HTTPException(status_code=400, detail=f"Invalid structure format: {e}. Problem near: '{section_str}'")
            else:
                # Fallback to single section using base parameters
                debug_print(2, "Using default structure (no structure parameter provided)")
                structure_data.append({
                    "length": params.length,
                    "density": params.density,
                    "instruments": default_included_drums,
                    "swing": default_preset_swing,
                    "apply_variations": not style_preset.get("use_strict_pattern", False),
                    "lead_instrument": None
                })
                total_beats = params.length
                debug_print(3, f"Default structure: {structure_data}")
                if not default_included_drums:
                     raise HTTPException(status_code=400, detail="No default instruments available for the selected style and include flags.")
        except HTTPException:
            # Let HTTP exceptions propagate
            raise
        except Exception as e:
            debug_print(1, f"ERROR in structure preparation: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error in structure preparation: {e}")
            
        # Debug output of final structure
        debug_print(1, f"Final structure data: {structure_data}")
        debug_print(1, f"Total beats: {total_beats}")

        # --- Generate Unique Filename ---
        unique_id = uuid.uuid4()
        base_filename = f"groove_{selected_style}_{params.bpm}bpm_{unique_id}"

        # --- Create Temporary Files Explicitly ---
        midi_temp_file = None
        wav_temp_file = None
        midi_output_path = None
        wav_output_path = None

        try: # Outer try for the whole generation process
            # Create named temporary files that won't be deleted automatically
            # Suffix is important for debugging and potential type detection
            # Use 'w+b' mode if needed, but saving functions might handle this
            midi_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
            wav_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            
            midi_output_path = Path(midi_temp_file.name)
            wav_output_path = Path(wav_temp_file.name)
            
            # Close the file handles immediately so other processes can write to them
            midi_temp_file.close()
            wav_temp_file.close()

            debug_print(1, f"Created temporary MIDI file: {midi_output_path}")
            debug_print(1, f"Created temporary WAV file: {wav_output_path}")

            # --- Call Groove Generation ---
            try:
                debug_print(1, "Calling generate_multi_drum_groove...")
                
                # Log important parameters
                debug_print(2, f"Style: {selected_style}")
                debug_print(2, f"BPM: {params.bpm}")
                debug_print(2, f"Structure (length): {len(structure_data)}")
                debug_print(2, f"Default included drums: {default_included_drums}")
                deep_perc = params.deep_perc_instruments.split(',') if params.deep_perc_instruments else None
                debug_print(2, f"Deep perc instruments: {deep_perc}")
                evolve = params.evolve_instruments.split(',') if params.evolve_instruments else None
                debug_print(2, f"Evolve instruments: {evolve}")
                
                midi_data = generate_multi_drum_groove(
                    style=selected_style,
                    bpm=params.bpm,
                    structure=structure_data,
                    default_included_drums=default_included_drums,
                    verbosity=1,
                    phrase_length_bars=params.phrase_length_bars,
                    deep_perc_instruments=deep_perc,
                    deep_perc_prob_start=params.deep_perc_prob_start,
                    deep_perc_prob_end=params.deep_perc_prob_end,
                    deep_perc_vel_start=params.deep_perc_vel_start,
                    deep_perc_vel_end=params.deep_perc_vel_end,
                    deep_perc_timing_ms_start=params.deep_perc_timing_ms_start,
                    deep_perc_timing_ms_end=params.deep_perc_timing_ms_end,
                    predictability=params.predictability,
                    evolve_instruments=evolve,
                    evolve_param=params.evolve_param,
                    evolve_start=params.evolve_start,
                    evolve_end=params.evolve_end,
                    adherence=params.adherence
                )
                debug_print(1, "MIDI data generation complete")
            except Exception as e:
                debug_print(1, f"ERROR in generate_multi_drum_groove: {e}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Error in MIDI generation: {e}")

            if not midi_data:
                raise HTTPException(status_code=500, detail="MIDI generation failed internally (returned None).")

            # --- Save MIDI File ---
            try:
                midi_data.save(midi_output_path)
                debug_print(1, f"MIDI file saved temporarily to: {midi_output_path}")
            except Exception as e:
                debug_print(1, f"Failed to save temporary MIDI file: {e}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Failed to save temporary MIDI file: {e}")

            # --- Prepare for WAV Rendering ---
            try:
                available_kits = {
                    "tr-909": "sounds/Roland TR-909",
                    "tr-808": "sounds/Roland TR-808"
                }
                selected_kit_name = params.kit.lower()
                primary_kit_path = available_kits.get(selected_kit_name)
                
                # Detailed validation of sound paths
                debug_print(1, f"Checking sound directory: {primary_kit_path}")
                sound_dir_exists = os.path.isdir(primary_kit_path) if primary_kit_path else False
                
                if not sound_dir_exists:
                    # Check if sounds/ exists at all
                    sounds_base_dir = "sounds"
                    if os.path.isdir(sounds_base_dir):
                        debug_print(1, f"Base sounds directory exists. Contents:")
                        for item in os.listdir(sounds_base_dir):
                            debug_print(1, f"  - {item}")
                            
                        # Try to look for any wav files
                        wav_files = glob.glob(f"{sounds_base_dir}/**/*.wav", recursive=True)
                        debug_print(1, f"Found {len(wav_files)} WAV files. First 5: {wav_files[:5]}")
                        
                        # Check for sound directories
                        subdirs = [f for f in os.listdir(sounds_base_dir) if os.path.isdir(os.path.join(sounds_base_dir, f))]
                        debug_print(1, f"Sound subdirectories: {subdirs}")
                    else:
                        debug_print(1, "Base sounds directory does not exist")
                    
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Selected kit '{params.kit}' path not found or invalid: {primary_kit_path}. Please check if sound directories are set up correctly."
                    )

                # Additional check for WAV files in the directory - use a case-insensitive pattern
                wav_files_upper = glob.glob(f"{primary_kit_path}/**/*.WAV", recursive=True)
                wav_files_lower = glob.glob(f"{primary_kit_path}/**/*.wav", recursive=True)
                wav_files = wav_files_upper + wav_files_lower
                debug_print(1, f"Found {len(wav_files)} WAV files in {primary_kit_path}. First 5: {wav_files[:5] if wav_files else 'none'}")
                
                if not wav_files:
                    raise HTTPException(status_code=400, detail=f"No WAV files found in the selected kit directory: {primary_kit_path}")

                # Determine fallback kit
                fallback_kit_path = None
                if selected_kit_name == "tr-909":
                    fallback_kit_path = available_kits.get("tr-808")
                elif selected_kit_name == "tr-808":
                     fallback_kit_path = available_kits.get("tr-909")
                if fallback_kit_path and not os.path.isdir(fallback_kit_path):
                    debug_print(1, f"[WARNING] Fallback kit path not found: {fallback_kit_path}")
                    fallback_kit_path = None # Disable fallback if path invalid

                # Load Samples
                debug_print(1, "Loading sample groups...")
                primary_sound_groups = group_samples_by_type(primary_kit_path, verbosity=0)
                debug_print(1, f"Primary sound groups: {primary_sound_groups.keys() if primary_sound_groups else 'none'}")
                
                fallback_sound_groups = None
                if fallback_kit_path:
                    fallback_sound_groups = group_samples_by_type(fallback_kit_path, verbosity=0)
                    debug_print(1, f"Fallback sound groups: {fallback_sound_groups.keys() if fallback_sound_groups else 'none'}")

                if not primary_sound_groups:
                    raise HTTPException(status_code=500, detail=f"Failed to load primary sound groups from: {primary_kit_path}")
            except Exception as e:
                debug_print(1, f"ERROR in sample preparation: {e}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Error in sample preparation: {e}")

            # --- Render WAV File ---
            try:
                debug_print(1, f"Rendering WAV to: {wav_output_path}")
                render_success = render_midi_with_sample_groups(
                    midi_file_path=str(midi_output_path), 
                    output_wav_path=str(wav_output_path), 
                    primary_sound_groups=primary_sound_groups,
                    fallback_sound_groups=fallback_sound_groups,
                    verbosity=0, 
                    panning_amount=params.panning,
                    humanize_amount=params.velocity_humanize,
                    style_preset=style_preset 
                )
                debug_print(1, f"WAV rendering complete: success={render_success}")
            except Exception as e:
                debug_print(1, f"ERROR in WAV rendering: {e}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Error in WAV rendering: {e}")

            # --- Verification ---
            if not render_success or not wav_output_path.exists() or os.path.getsize(wav_output_path) <= 0:
                debug_print(1, f"WAV rendering failed, file not found, or file is empty at: {wav_output_path}")
                if midi_output_path.exists():
                    debug_print(1, f"MIDI file exists at: {midi_output_path}")
                else:
                    debug_print(1, f"MIDI file also does not exist at: {midi_output_path}")
                raise HTTPException(status_code=500, detail=f"WAV rendering failed. Expected file: {wav_output_path}")

            # --- Return WAV File --- 
            debug_print(1, f"Returning WAV file: {wav_output_path}")
            # Add cleanup tasks for BOTH temporary files AFTER the response is sent
            background_tasks.add_task(cleanup_file, wav_output_path)
            background_tasks.add_task(cleanup_file, midi_output_path) 

            return FileResponse(
                path=wav_output_path,
                filename=f"{base_filename}.wav",
                media_type='audio/wav'
            )
        except HTTPException:
            # Let HTTP exceptions propagate
            raise
        except Exception as e:
            debug_print(1, f"ERROR in sample preparation: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error in sample preparation: {e}")

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        debug_print(1, f"HTTP exception occurred: {http_exc.detail}")
        # Clean up any created temp files if an HTTP exception occurs mid-process
        if midi_output_path and os.path.exists(midi_output_path):
            cleanup_file(midi_output_path)
        if wav_output_path and os.path.exists(wav_output_path):
            cleanup_file(wav_output_path)
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors
        debug_print(1, f"Unexpected error during generation process: {e}")
        traceback.print_exc()
        # Clean up any created temp files on general error
        if midi_output_path and os.path.exists(midi_output_path):
            cleanup_file(midi_output_path)
        if wav_output_path and os.path.exists(wav_output_path):
            cleanup_file(wav_output_path)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# --- Add a function to test/debug the generation ----
def debug_generate():
    """Test the generation function directly with minimal parameters"""
    try:
        print("-"*50)
        print("DEBUG: Testing generate_multi_drum_groove with minimal parameters")
        
        # Create a minimal structure
        test_structure = [{
            "length": 16,  # Length in beats
            "density": "medium",
            "instruments": ["kick", "snare", "closed_hat", "open_hat"],
            "swing": 0.0,
            "apply_variations": True,
            "lead_instrument": None
        }]
        
        print(f"Test structure: {test_structure}")
        
        # Call the function
        result = generate_multi_drum_groove(
            style="dusty",
            bpm=90,
            structure=test_structure,
            default_included_drums=["kick", "snare", "closed_hat", "open_hat"],
            verbosity=1,
            phrase_length_bars=4,
            deep_perc_instruments=None,
            deep_perc_prob_start=0.1,
            deep_perc_prob_end=0.5,
            deep_perc_vel_start=70,
            deep_perc_vel_end=110,
            deep_perc_timing_ms_start=15.0,
            deep_perc_timing_ms_end=3.0,
            predictability=0.5,
            evolve_instruments=None,
            evolve_param="probability",
            evolve_start=0.0,
            evolve_end=1.0,
            adherence=0.0
        )
        
        print(f"Result: {result}")
        print("-"*50)
        return result
    except Exception as e:
        print(f"ERROR in debug_generate: {e}")
        import traceback
        traceback.print_exc()
        print("-"*50)
        return None

# Add a wrapped version of _should_place_note that validates pattern_weights
def safe_should_place_note(style_preset, drum_preset, drum_type, beat, step, only_kicks, density_factor, **kwargs):
    """
    Wrapper for _should_place_note that validates pattern_weights to prevent list index out of range errors.
    """
    # Check if pattern_weights exists and has enough elements
    if not drum_preset:
        debug_print(1, f"WARNING: Missing drum_preset for {drum_type}, returning False")
        return False
        
    # Make a copy of the drum preset we can modify if needed
    safe_drum_preset = drum_preset.copy()
    
    # Handle missing or too-short pattern_weights
    pattern_weights = safe_drum_preset.get("pattern_weights", None)
    if pattern_weights is None:
        debug_print(1, f"WARNING: Missing pattern_weights for {drum_type}, using default [1.0] * 16")
        safe_drum_preset["pattern_weights"] = [1.0] * 16
    elif len(pattern_weights) < 16:
        debug_print(1, f"WARNING: pattern_weights for {drum_type} has only {len(pattern_weights)} elements, extending to 16")
        # Extend pattern_weights to 16 elements by repeating it
        current_length = len(pattern_weights)
        if current_length > 0:
            # Calculate how many times to repeat the existing pattern
            repeat_times = (16 + current_length - 1) // current_length  # Ceiling division
            safe_drum_preset["pattern_weights"] = (pattern_weights * repeat_times)[:16]
        else:
            safe_drum_preset["pattern_weights"] = [1.0] * 16
    
    try:
        # Call the original function with the safe drum preset
        return _should_place_note(style_preset, safe_drum_preset, drum_type, beat, step, only_kicks, density_factor, **kwargs)
    except Exception as e:
        debug_print(1, f"ERROR in _should_place_note for {drum_type}: {e}")
        return False

# Now replace all calls to _should_place_note with safe_should_place_note in the generate_multi_drum_groove
# To do this without modifying the original function, we'll monkey patch it in the startup_event

@app.on_event("startup")
async def startup_event():
    """Run when the app starts"""
    try:
        # Save the original function for reference
        global original_should_place_note
        original_should_place_note = getattr(sys.modules['groove_agent_multi'], '_should_place_note')
        
        # Replace the original function with our safe version
        setattr(sys.modules['groove_agent_multi'], '_should_place_note', safe_should_place_note)
        
        # Also set a reference to the original function in our safe version
        setattr(sys.modules[__name__], '_should_place_note', original_should_place_note)
        
        print("="*50)
        print("STARTUP: Monkey patched _should_place_note with safe version")
        print("="*50)
        
        # Run the debug test
        result = debug_generate()
        print("="*50)
        print(f"DEBUG GENERATE RESULT: {'Success' if result else 'Failed'}")
        print("="*50)
    except Exception as e:
        print("="*50)
        print(f"ERROR IN STARTUP: {e}")
        import traceback
        traceback.print_exc()
        print("="*50)

# --- To Run (from terminal) ---
# uvicorn main:app --reload 
