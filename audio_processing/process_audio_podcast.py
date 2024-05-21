import argparse
import wave
import numpy as np
import os
import re
import shutil
import json

def read_audio_file(file_path):
    with wave.open(file_path, 'r') as audio_file:
        params = audio_file.getparams()
        rate = audio_file.getframerate()
        nframes = audio_file.getnframes()
        frames = audio_file.readframes(nframes)
        audio_data = np.frombuffer(frames, dtype=np.int16)
    return params, rate, audio_data

def create_silence(length, n_channels):
    return np.zeros(length * n_channels, dtype=np.int16)

def get_audio_duration(file_path):
    with wave.open(file_path, 'r') as audio_file:
        frames = audio_file.getnframes()
        rate = audio_file.getframerate()
        duration = frames / float(rate)
    return duration

# Function to create subtitle JSON files for each slide
def create_subtitle_files(podcast_data, audio_directory, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    slide_dict = {}

    # Group subtitles by slide and get audio durations
    for entry in podcast_data:
        slide = entry["slide"]
        question = entry["question"]
        answer = entry["answer"]
        
        if slide not in slide_dict:
            slide_dict[slide] = []

        # Find corresponding audio files
        question_audio_file = os.path.join(audio_directory, f"slide{slide}-host.wav")
        answer_audio_file = os.path.join(audio_directory, f"slide{slide}-guest.wav")

        # Get the duration from the audio files
        question_duration = get_audio_duration(question_audio_file)
        answer_duration = get_audio_duration(answer_audio_file)

        # Add question and answer to the slide entry
        slide_dict[slide].append({
            "subtitle": question,
            "duration": question_duration
        })
        slide_dict[slide].append({
            "subtitle": answer,
            "duration": answer_duration
        })

    # Write each slide's subtitles to a separate JSON file
    for slide, subtitles in slide_dict.items():
        output_file_path = os.path.join(output_dir, f"{slide}.json")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(subtitles, f, ensure_ascii=False, indent=4)
        print(f"Created file: {output_file_path}")

def process_and_merge_audio(slides_order, guest_files, host_files, output_dir):
    guest_combined_data = np.array([], dtype=np.int16)
    host_combined_data = np.array([], dtype=np.int16)
    n_channels = None

    for slide in slides_order:
        host_file = host_files.get(slide)
        guest_file = guest_files.get(slide)

        if host_file is None or guest_file is None:
            print(f"Missing files for slide {slide}, skipping.")
            continue

        print(f"Processing slide {slide} with host file {host_file} and guest file {guest_file}")
        
        # Read host data
        host_params, host_rate, host_data = read_audio_file(host_file)
        if n_channels is None:
            n_channels = host_params.nchannels
        host_silence = create_silence(len(host_data) // host_params.nchannels, host_params.nchannels)

        # Read guest data
        guest_params, guest_rate, guest_data = read_audio_file(guest_file)
        guest_silence = create_silence(len(guest_data) // guest_params.nchannels, guest_params.nchannels)

        assert host_rate == guest_rate, "Sample rates do not match"

        # Host talks first, then guest
        host_segment = np.concatenate((host_data, guest_silence))
        guest_segment = np.concatenate((host_silence, guest_data))

        host_combined_data = np.concatenate((host_combined_data, host_segment))
        guest_combined_data = np.concatenate((guest_combined_data, guest_segment))

    # Write combined data to output files
    if len(guest_files) > 0:
        guest_params, _, _ = read_audio_file(next(iter(guest_files.values())))
        guest_output_file = os.path.join(output_dir, "guest.wav")
        with wave.open(guest_output_file, 'w') as output:
            output.setparams(guest_params)
            output.writeframes(guest_combined_data.tobytes())

    if len(host_files) > 0:
        host_params, _, _ = read_audio_file(next(iter(host_files.values())))
        host_output_file = os.path.join(output_dir, "host.wav")
        with wave.open(host_output_file, 'w') as output:
            output.setparams(host_params)
            output.writeframes(host_combined_data.tobytes())

    print("Processed and merged audio files created.")

def gather_files_from_directory(directory):
    guest_files = {}
    host_files = {}

    for filename in os.listdir(directory):
        match = re.match(r'slide(\d+)-(guest|host)\.wav', filename)
        if match:
            slide_number = int(match.group(1))
            role = match.group(2)
            file_path = os.path.join(directory, filename)

            if role == 'guest':
                guest_files[slide_number] = file_path
            elif role == 'host':
                host_files[slide_number] = file_path

    return guest_files, host_files

def move_files_to_audio_folder(files, audio_folder):
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)
    for file in files:
        shutil.move(file, audio_folder)

def main(audio_directory, output_dir, script_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(script_file, 'r') as file:
        script = json.load(file)

    slides_order = [item["slide"] for item in script]
    guest_files, host_files = gather_files_from_directory(audio_directory)
    process_and_merge_audio(slides_order, guest_files, host_files, output_dir)

    audio_folder = os.path.join(output_dir, 'audio')
    move_files_to_audio_folder(list(guest_files.values()) + list(host_files.values()), audio_folder)
    print(f"Moved all audio files to {audio_folder}")

        # Load the podcast JSON file
    with open(script_file, 'r', encoding='utf-8') as f:
        podcast_data = json.load(f)

    # Create subtitle JSON files for each slide
    create_subtitle_files(podcast_data, audio_folder, os.path.join(output_dir, 'subtitle'))

    # Remove the audio folder
    shutil.rmtree(audio_folder)
    print(f"Removed the audio folder {audio_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and merge audio files for podcast")
    parser.add_argument('--audio_directory', type=str, required=True, help="Directory containing the audio files")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the processed audio files")
    parser.add_argument('--script_file', type=str, required=True, help="JSON file containing the podcast script")
    args = parser.parse_args()
    main(args.audio_directory, args.output_dir, args.script_file)



# data/podcast/podcast_script.json
