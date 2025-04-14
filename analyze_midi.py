from mido import MidiFile
import sys

def analyze_midi(midi_file):
    mid = MidiFile(midi_file)
    
    print(f"File: {midi_file}")
    print(f"Format: {mid.type}")
    print(f"Tracks: {len(mid.tracks)}")
    print(f"Ticks per beat: {mid.ticks_per_beat}")
    
    total_ticks = 0
    note_events = 0
    
    for i, track in enumerate(mid.tracks):
        print(f"\nTrack {i}:")
        
        track_ticks = 0
        track_notes = 0
        track_notes_on = 0
        track_notes_off = 0
        notes_by_pitch = {}
        
        for msg in track:
            track_ticks += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                track_notes += 1
                track_notes_on += 1
                notes_by_pitch[msg.note] = notes_by_pitch.get(msg.note, 0) + 1
                print(f"  {track_ticks} ticks: Note ON  {msg.note} velocity={msg.velocity}")
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                track_notes_off += 1
                print(f"  {track_ticks} ticks: Note OFF {msg.note}")
            elif msg.is_meta:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    bpm = 60_000_000 / tempo
                    print(f"  {track_ticks} ticks: Tempo change to {bpm} BPM")
                elif msg.type == 'time_signature':
                    print(f"  {track_ticks} ticks: Time Signature {msg.numerator}/{msg.denominator}")
                elif msg.type == 'end_of_track':
                    print(f"  {track_ticks} ticks: End of Track")
        
        total_ticks = max(total_ticks, track_ticks)
        note_events += track_notes
        
        print(f"  Track duration: {track_ticks} ticks")
        print(f"  Notes: {track_notes} ({track_notes_on} on, {track_notes_off} off)")
        
        if notes_by_pitch:
            print("  Notes by pitch:")
            for pitch, count in sorted(notes_by_pitch.items()):
                print(f"    MIDI note {pitch}: {count} hits")
    
    # Calculate approximate duration in seconds
    if hasattr(mid, 'ticks_per_beat') and mid.ticks_per_beat > 0:
        # Assuming a tempo of 120 BPM if not specified
        seconds_per_beat = 60 / 120
        seconds_per_tick = seconds_per_beat / mid.ticks_per_beat
        duration_seconds = total_ticks * seconds_per_tick
        
        # Convert to minutes and seconds
        minutes = int(duration_seconds // 60)
        seconds = duration_seconds % 60
        
        print(f"\nEstimated duration: {minutes}m {seconds:.2f}s ({total_ticks} ticks)")
    
    print(f"Total note events: {note_events}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_midi.py <midi_file>")
        sys.exit(1)
    
    analyze_midi(sys.argv[1]) 
