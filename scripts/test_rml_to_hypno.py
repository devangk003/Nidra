import os
import xml.etree.myElementTree as ET
import numpy as np
import matplotlib.pyplot as plt


def parse_rml(file_path):
    """Parse the .rml file and extract sleep stage data."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        stages = []
        for stage in root.findall("Stage"):
            stage_type = stage.get("Type")
            start_time = int(stage.get("Start"))
            stages.append((stage_type, start_time))
        return stages
    except ET.ParseError as e:
        print(f"[ERROR] Failed to parse the file: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while parsing the file: {e}")
        raise


def create_hypnogram(stages):
    """Convert stage data into a hypnogram array."""
    # Define stage mappings (Reversed to match Wake at the top)
    stage_map = {
        "Wake": 0,  # Now Wake is 0 (top)
        "REM": 1,
        "NonREM1": 2,  # Light Sleep
        "NonREM2": 2,  # Light Sleep
        "NonREM3": 3,  # Deep Sleep (bottom)
    }

    # Sort stages by their start time
    stages = sorted(stages, key=lambda x: x[1])

    # Determine the duration of the recording
    recording_duration = stages[-1][1] + 30  # Assuming the final stage lasts at least 30 seconds

    # Initialize the hypnogram with default stage (Wake)
    hypnogram = np.full(recording_duration, stage_map["Wake"])

    for i in range(len(stages) - 1):
        current_stage = stage_map.get(stages[i][0], stage_map["Wake"])  # Default to Wake if unknown
        start_time = stages[i][1]
        end_time = stages[i + 1][1]
        hypnogram[start_time:end_time] = current_stage

    # Handle the last stage
    last_stage = stage_map.get(stages[-1][0], stage_map["Wake"])
    hypnogram[stages[-1][1]:] = last_stage

    return hypnogram


def plot_hypnogram(hypnogram):
    """Plot the hypnogram with reversed y-axis and adjusted values."""
    # Define stage labels (Reversed order to match Wake at the top)
    stage_labels = {
        0: "Wake",
        1: "REM",
        2: "Light Sleep",
        3: "Deep Sleep",
    }

    # Flip the hypnogram data
    hypnogram = np.max(list(stage_labels.keys())) - hypnogram

    # Generate time in minutes
    time_in_minutes = np.arange(len(hypnogram)) / 60

    plt.figure(figsize=(12, 6))
    plt.step(time_in_minutes, hypnogram, where="post", color="blue", linewidth=2)
    
    # Reverse the y-axis labels order to match the new data order
    plt.yticks(list(stage_labels.keys())[::-1], list(stage_labels.values())[::-1])

    plt.xlabel("Time (minutes)", fontsize=14)
    plt.ylabel("Sleep Stage", fontsize=14)
    plt.title("Sleep Hypnogram", fontsize=16)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def main():
    # Replace this with the actual path to your .rml file
    rml_file = r"E:\Nidra\dataset\APNEA_RML\00001008-100507\00001008-100507.rml"  # Use a raw string for Windows paths

    # Check if the file exists
    if not os.path.exists(rml_file):
        print(f"[ERROR] File not found: {rml_file}")
        return

    try:
        # Parse the .rml file
        stages = parse_rml(rml_file)
        print("[INFO] Stages extracted successfully.")

        # Create a hypnogram
        hypnogram = create_hypnogram(stages)
        print("[INFO] Hypnogram created successfully.")

        # Plot the hypnogram
        plot_hypnogram(hypnogram)
        print("[INFO] Hypnogram plotted successfully.")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")


if __name__ == "__main__":
    main()
