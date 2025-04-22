import os
import wave
import re
import sys

# Regex pattern with underscore match
pattern = re.compile(r'^(.+?)\[(\d+)\]_(\w+)\.wav$')

def are_params_compatible(p1, p2):
    """Check if parameters are mergeable (ignore nframes)"""
    return (p1.nchannels == p2.nchannels and
            p1.sampwidth == p2.sampwidth and
            p1.framerate == p2.framerate and
            p1.comptype == p2.comptype)

# Dictionary to group files by (base, type)
file_groups = {}

print("Current working directory:", os.getcwd())
print("\nFiles in directory:")
for f in os.listdir('.'):
    print("-", f)

# Scan current directory for WAV files
for filename in os.listdir('.'):
    if not filename.endswith('.wav'):
        continue
    
    match = pattern.match(filename)
    if not match:
        print(f"\nSkipped {filename} (didn't match pattern)")
        continue
    
    base_name = match.group(1)
    serial = int(match.group(2))
    file_type = match.group(3)
    
    print(f"\nMatched file: {filename}")
    print(f"Base: '{base_name}' | Serial: {serial} | Type: {file_type}")
    
    key = (base_name, file_type)
    if key not in file_groups:
        file_groups[key] = []
    file_groups[key].append((serial, filename))

if not file_groups:
    print("\nNo valid files found matching the pattern!")
    sys.exit(1)

# Process each group of files
for (base, file_type), files in file_groups.items():
    print(f"\nProcessing group: {base}_{file_type}")
    sorted_files = sorted(files, key=lambda x: x[0])
    output_filename = f"{base}_{file_type}.wav"
    
    print(f"Output file will be: {output_filename}")
    print(f"Files to combine ({len(sorted_files)}):")
    for f in sorted_files:
        print(f"- {f[1]}")

    params = None
    frames = bytearray()
    success = True
    
    try:
        for i, (serial, filename) in enumerate(sorted_files):
            with wave.open(filename, 'rb') as wav:
                current_params = wav.getparams()
                print(f"\nFile {i+1}: {filename}")
                print(f"Params: {current_params}")
                
                if not params:
                    params = current_params
                else:
                    if not are_params_compatible(params, current_params):
                        print(f"PARAMETER MISMATCH! Previous: {params} | Current: {current_params}")
                        success = False
                        break
                
                frames.extend(wav.readframes(wav.getnframes()))
        
        if success:
            with wave.open(output_filename, 'wb') as output:
                output.setparams(params)
                output.writeframes(frames)
            print(f"\nSuccessfully created: {output_filename}")
            print(f"Combined duration: {len(frames)/params.sampwidth/params.nchannels/params.framerate:.2f} seconds")
        else:
            print(f"Skipped {output_filename} due to parameter mismatch")
            
    except Exception as e:
        print(f"ERROR processing {output_filename}: {str(e)}")
        continue

print("\nProcessing complete!")
print("Final files in directory:")
for f in os.listdir('.'):
    print("-", f)