import sys
import argparse
from mido import MidiFile
import math

def visualize_midi_drums(midi_file_path, bars=4):
    # Read MIDI file
    mid = MidiFile(midi_file_path)
    ticks_per_beat = mid.ticks_per_beat
    
    # Collect all note_on events with absolute time
    note_events = []
    current_tick = 0
    
    for track in mid.tracks:
        track_tick = 0
        for msg in track:
            track_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                note_events.append((track_tick, msg.note, msg.velocity))
    
    if not note_events:
        print("No notes found in the MIDI file.")
        return
    
    # Sort events by time
    note_events.sort()
    
    # Calculate total duration and grid width
    max_tick = note_events[-1][0]
    beats = math.ceil(max_tick / ticks_per_beat)
    grid_width = min(beats * 4, bars * 16)  # 4 16ths per beat
    
    # Create a grid
    drums = {
        36: "K",  # Kick
        38: "S",  # Snare
        42: "H",  # Closed Hat
        46: "O"   # Open Hat
    }
    
    # Initialize grid
    grid = {}
    for note in drums:
        grid[note] = [" " for _ in range(grid_width)]
    
    # Fill grid with notes
    for tick, note, velocity in note_events:
        if note in drums:
            grid_pos = int((tick / ticks_per_beat) * 4)  # Convert to 16th notes
            if grid_pos < grid_width:
                # Use velocity to show intensity
                if velocity > 100:
                    grid[note][grid_pos] = drums[note]
                else:
                    grid[note][grid_pos] = drums[note].lower()
    
    # Print header
    print("\nMIDI Drum Pattern Visualization:")
    print("-------------------------------")
    
    # Calculate how many complete bars we'll show
    display_bars = min(math.ceil(grid_width / 16), bars)
    
    # Print bar and beat numbers
    bar_header = "    "
    beat_header = "    "
    for bar in range(1, display_bars + 1):
        for beat in range(1, 5):
            bar_header += f"Bar {bar}     "
            beat_header += f"Beat {beat}   "
    print(bar_header)
    print(beat_header)
    
    # Print grid markers (every 16th note)
    beat_markers = "    "
    for i in range(grid_width):
        if i % 4 == 0:
            beat_markers += "|"  # Beat marker
        else:
            beat_markers += "."  # Subdivision marker
        beat_markers += " "
    print(beat_markers)
    
    # Print grid for each drum type
    for note in sorted(drums.keys()):
        symbol = drums[note]
        row = f"{symbol}: "
        for i in range(grid_width):
            row += grid[note][i] + " "
        print(row)
    
    print("-------------------------------")
    print("K=Kick, S=Snare, H=Closed Hat, O=Open Hat")
    print(f"Total notes: {len(note_events)}")
    print(f"Note distribution: {note_count_by_type(note_events, drums)}")

def note_count_by_type(note_events, drums):
    """Count notes by type"""
    counts = {}
    for _, note, _ in note_events:
        if note in drums:
            drum_type = drums[note]
            counts[drum_type] = counts.get(drum_type, 0) + 1
    return counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MIDI drum patterns in the console")
    parser.add_argument("midi_file", help="Path to the MIDI file to visualize")
    parser.add_argument("--bars", type=int, default=4, help="Number of bars to visualize (default: 4)")
    
    args = parser.parse_args()
    visualize_midi_drums(args.midi_file, args.bars) 
