import os
import cv2
import torch.nn as nn
import torch
import sys
import json
import shutil
import subprocess
import io
import tempfile
from glob import glob
from pydub import AudioSegment
from argparse import ArgumentParser
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from concurrent.futures import ThreadPoolExecutor
import psutil


def check_disk_usage(tempdir):
    disk_usage = psutil.disk_usage(tempdir)
    print(f"Disk usage for {tempdir}:")
    print(f"  Total: {disk_usage.total / (1024 ** 3):.2f} GB")
    print(f"  Used: {disk_usage.used / (1024 ** 3):.2f} GB")
    print(f"  Free: {disk_usage.free / (1024 ** 3):.2f} GB")
    print(f"  Percent: {disk_usage.percent}%")

def check_memory_usage():
    memory_info = psutil.virtual_memory()
    print("Memory usage:")
    print(f"  Total: {memory_info.total / (1024 ** 3):.2f} GB")
    print(f"  Available: {memory_info.available / (1024 ** 3):.2f} GB")
    print(f"  Used: {memory_info.used / (1024 ** 3):.2f} GB")
    print(f"  Percent: {memory_info.percent}%")



def chunk_audio_sequential(file_path, chunk_length_ms):
    audio = AudioSegment.from_file(file_path)
    audio_length_ms = len(audio)

    for i in range(0, audio_length_ms, chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_io = io.BytesIO()
        chunk.export(chunk_io, format="wav")
        yield chunk_io.getvalue()


def convert_audio_to_video(audio_path, pic_path, save_dir, args):
    aux_main(audio_path, pic_path, save_dir, args)
    return os.path.join("", save_dir + '.mp4')

def overlay_video_on_image(input_image_path, face_positions, video_paths, output_path, tempdir):
    input_image = cv2.imread(input_image_path)
    height, width, _ = input_image.shape

    filter_complex = []
    inputs = ['-i', input_image_path]
    for i, video_path in enumerate(video_paths):
        inputs += ['-i', video_path]
        pos = face_positions[i]
        x, y, w, h = pos['left'], pos['top'], pos['right'] - pos['left'], pos['bottom'] - pos['top']
        scale_filter = f"[{i+1}:v]scale={w}:{h}[video{i}]"
        filter_complex.append(scale_filter)
        if i == 0:
            overlay_filter = f"[0:v][video{i}]overlay={x}:{y}[tmp{i+1}]"
        else:
            overlay_filter = f"[tmp{i}][video{i}]overlay={x}:{y}[tmp{i+1}]"
        filter_complex.append(overlay_filter)

    filter_complex[-1] = filter_complex[-1].replace(f"[tmp{len(video_paths)}]", "[v]")

    filter_complex = ';'.join(filter_complex)

    temp_video_path = os.path.join(tempdir, "temp_video.mp4")

    cmd = [
            'ffmpeg',
            *inputs,
            '-filter_complex', filter_complex,
            '-map', '[v]',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            temp_video_path,
            '-y'
        ]
    run_ffmpeg_command(cmd)

    combined_audio_path = os.path.join(tempdir, 'combined_audio.wav')
    inputs = ['-i', video_paths[0]]
    for video_path in video_paths[1:]:
        inputs += ['-i', video_path]

    audio_filter_complex = '[0:a]aresample=async=1:first_pts=0[a0];'
    for i in range(1, len(video_paths)):
        audio_filter_complex += f'[{i}:a]aresample=async=1:first_pts=0[a{i}];'
    audio_filter_complex += ''.join([f'[a{i}]' for i in range(len(video_paths))])
    audio_filter_complex += f'amix=inputs={len(video_paths)}:duration=longest[a]'

    cmd = [
        'ffmpeg',
        *inputs,
        '-filter_complex', audio_filter_complex,
        '-map', '[a]',
        combined_audio_path,
        '-y'
    ]
    run_ffmpeg_command(cmd)

    cmd = [
        'ffmpeg',
        '-i', temp_video_path,
        '-i', combined_audio_path,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:a', 'aac',
        output_path,
        '-y'
    ]
    run_ffmpeg_command(cmd)

    os.remove(temp_video_path)
    os.remove(combined_audio_path)

    check_disk_usage(tempdir)
    check_memory_usage()




def convert_audio_chunk_to_video(audio_chunk, pic_path, fallback_pic_path, args, tempdir):
    with tempfile.NamedTemporaryFile(suffix=".wav", dir=tempdir, delete=False) as temp_audio_file:
        temp_audio_file.write(audio_chunk)
        temp_audio_file.flush()
        audio_path = temp_audio_file.name
        video_path = os.path.join(tempdir, os.path.splitext(os.path.basename(audio_path))[0])
        os.makedirs(video_path, exist_ok=True)
        try:
            video_file = convert_audio_to_video(audio_path, pic_path, video_path, args)
        except Exception as e:
            print(f"Error detected during video conversion: {e}. Falling back to original image for {audio_path}")
            video_file = convert_audio_to_video(audio_path, fallback_pic_path, video_path, args)
    os.remove(audio_path)  # Remove the temporary audio file after use
    check_disk_usage(tempdir)
    check_memory_usage()
    return video_file


def concatenate_videos(video_path1, video_path2, output_path):
    temp_output_path = output_path + ".temp.mp4"

    cmd = [
        'ffmpeg',
        '-i', video_path1,
        '-i', video_path2,
        '-filter_complex', '[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[outv][outa]',
        '-map', '[outv]',
        '-map', '[outa]',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        temp_output_path,
        '-y'
    ]

    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    run_ffmpeg_command(cmd)
    shutil.move(temp_output_path, output_path)



def extract_last_frame(video_path, tempdir, role):
    print("Extracting last frame from video path: ", video_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        print(f"Video file size: {os.path.getsize(video_path)} bytes")
        raise RuntimeError(f"Failed to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"No frames found in video file: {video_path}")
        raise RuntimeError(f"No frames found in video file: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read the last frame from video file: {video_path}")
        raise RuntimeError(f"Failed to read the last frame from video file: {video_path}")

    last_frame_path = os.path.join(tempdir, f"last_frame_{role}.png")
    cv2.imwrite(last_frame_path, frame)
    cap.release()

    return last_frame_path




def process_and_append_chunks(audio_path_guest, audio_path_host, pic_path_guest, pic_path_host, face_positions, input_image_path, args, tempdir, final_video_path):
    chunk_length_ms = args.chunk_length_ms  # Use the value from args

    guest_audio_gen = chunk_audio_sequential(audio_path_guest, chunk_length_ms)
    host_audio_gen = chunk_audio_sequential(audio_path_host, chunk_length_ms)

    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=tempdir, delete=False) as temp_final_video:
        final_video_temp_path = temp_final_video.name

    with ThreadPoolExecutor() as executor:
        chunk_index = 0
        prev_guest_pic_path = pic_path_guest
        prev_host_pic_path = pic_path_host

        while True:
            guest_future = None
            host_future = None

            try:
                guest_chunk = next(guest_audio_gen)
                guest_future = executor.submit(convert_audio_chunk_to_video, guest_chunk, prev_guest_pic_path, pic_path_guest, args, tempdir)
            except StopIteration:
                guest_chunk = None

            try:
                host_chunk = next(host_audio_gen)
                host_future = executor.submit(convert_audio_chunk_to_video, host_chunk, prev_host_pic_path, pic_path_host, args, tempdir)
            except StopIteration:
                host_chunk = None

            if guest_future is None and host_future is None:
                break

            video_files = []
            if guest_future:
                video_files.append(guest_future.result())
            if host_future:
                video_files.append(host_future.result())

            if len(video_files) > 0:
                temp_overlay_video_path = os.path.join(tempdir, f"temp_overlay_chunk_{chunk_index}.mp4")
                overlay_video_on_image(input_image_path, face_positions, video_files, temp_overlay_video_path, tempdir)

                if chunk_index == 0:
                    shutil.copy(temp_overlay_video_path, final_video_temp_path)
                else:
                    temp_concatenated_video_path = os.path.join(tempdir, f"temp_concatenated_{chunk_index}.mp4")
                    concatenate_videos(final_video_temp_path, temp_overlay_video_path, temp_concatenated_video_path)
                    os.remove(final_video_temp_path)  # Clean up previous temp file
                    final_video_temp_path = temp_concatenated_video_path

                # Clean up temporary video files
                for video_path in video_files:
                    os.remove(video_path)

                os.remove(temp_overlay_video_path)
                chunk_index += 1

    shutil.move(final_video_temp_path, final_video_path)
    check_disk_usage(tempdir)
    check_memory_usage()



def resize_image(image_path, width, height):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite(image_path, resized_image)

def aux_main(audio_path, pic_path, save_dir, args):
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)

    current_root_path = os.path.split(sys.argv[0])[0]
    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size, args.old_version, args.preprocess)

    preprocess_model = CropAndExtract(sadtalker_paths, args.device)
    audio_to_coeff = Audio2Coeff(sadtalker_paths, args.device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, args.device)


    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for parallel processing.")
        preprocess_model = nn.DataParallel(preprocess_model, device_ids=[1, 2, 3])
        audio_to_coeff = nn.DataParallel(audio_to_coeff, device_ids=[1, 2, 3])
        animate_from_coeff = nn.DataParallel(animate_from_coeff, device_ids=[1, 2, 3])
    else:
        print("Using a single GPU.")
    preprocess_model.to(args.device)
    audio_to_coeff.to(args.device)
    animate_from_coeff.to(args.device)

    print('3DMM Extraction for source image')
    if torch.cuda.device_count() > 1:
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.module.generate(pic_path, first_frame_dir, args.preprocess, source_image_flag=True, pic_size=args.size)
    else:
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir, args.preprocess, source_image_flag=True, pic_size=args.size)
    
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return None

    batch = get_data(first_coeff_path, audio_path, args.device, None, still=args.still)
    if torch.cuda.device_count() > 1:
        coeff_path = audio_to_coeff.module.generate(batch, save_dir, args.pose_style, None)
    else:
        coeff_path = audio_to_coeff.generate(batch, save_dir, args.pose_style, None)

    if args.face3dvis:
        from SadTalker.src.face3d.visualize import gen_composed_video
        gen_composed_video(args, args.device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))

    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, args.batch_size, args.input_yaw, args.input_pitch, args.input_roll, expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)
    

    if torch.cuda.device_count() > 1:
        result = animate_from_coeff.module.generate(data, save_dir, pic_path, crop_info, enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess, img_size=args.size)
    else:
        result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess, img_size=args.size)
    
    final_video_path = save_dir + '.mp4'
    shutil.move(result, final_video_path)
    print('The generated video is named:', final_video_path)

    if not args.verbose:
        shutil.rmtree(save_dir)

    return final_video_path

def find_label_file(source_image_path):
    base_output_dir = "examples/source_image/output"
    base_name = os.path.basename(source_image_path)
    name, _ = os.path.splitext(base_name)
    label_pattern = os.path.join(base_output_dir, 'exp*', 'labels', f"{name}.txt")
    label_files = glob(label_pattern)

    if label_files:
        return label_files[0]
    else:
        raise FileNotFoundError(f"No label text file found for image {source_image_path}")

def read_labels(input_image_path, label_file_path, border=0):
    input_image = cv2.imread(input_image_path)
    img_height, img_width, _ = input_image.shape
    face_positions = []
    with open(label_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            x_center, y_center, width, height = map(float, parts)
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            left = int(x_center - width / 2) - border
            top = int(y_center - height / 2) - border
            right = int(x_center + width / 2) + border
            bottom = int(y_center + width / 2) + border
            left = max(0, left)
            top = max(0, top)
            right = min(img_width, right)
            bottom = min(img_height, bottom)
            face_positions.append({'left': left, 'top': top, 'right': right, 'bottom': bottom})
    return face_positions

def extract_faces(input_image_path, face_positions, output_dir):
    image = cv2.imread(input_image_path)
    face_image_paths = []
    audio_mapping = {}

    for i, pos in enumerate(face_positions):
        left, top, right, bottom = pos['left'], pos['top'], pos['right'], pos['bottom']
        face_image = image[top:bottom, left:right]
        face_image_name = f"face_{i+1}.png"
        face_image_path = os.path.join(output_dir, face_image_name)
        cv2.imwrite(face_image_path, face_image)
        face_image_paths.append(face_image_path)
        print("Path : ", face_image_paths)
        
        # Modify based on your file wav
        if i == 0:
            audio_mapping[face_image_name] = "guest.wav" # host.wav | host_small.wav for short clip
        elif i == 1:
            audio_mapping[face_image_name] = "host.wav" # guest.wav | guest_small.wav for short clip
        else:
            # Code for lecture must have one face
            raise ValueError("Podcast video must have exactly two faces")
    
    mapping_file_path = os.path.join(output_dir, "audio_mapping.json")

    # Return the contents of the audio_mapping.json file if it exists
    if os.path.exists(mapping_file_path):
        with open(mapping_file_path, 'r') as f:
            existing_audio_mapping = json.load(f)
            print(f"Audio mapping JSON loaded from: {mapping_file_path}")
            return face_image_paths, existing_audio_mapping
    else:
        # Write the dictionary to a JSON file if it does not exist
        with open(mapping_file_path, 'w') as f:
            json.dump(audio_mapping, f, indent=4)
            print(f"Audio mapping JSON saved at: {mapping_file_path}")
        return face_image_paths, audio_mapping

def run_ffmpeg_command(cmd):
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("FFmpeg output:", result.stdout.decode())
        print("FFmpeg error (stderr):", result.stderr.decode())
    except subprocess.CalledProcessError as e:
        print("Error while running command: ", " ".join(e.cmd))
        print("Standard output: ", e.stdout.decode())
        print("Standard error: ", e.stderr.decode())
        raise




def main():
    parser = ArgumentParser()  
    parser.add_argument("--driven_audio_folder", default='./examples/driven_audio', help="path to folder containing driven audio files")    
    parser.add_argument("--source_image", default='./data/podcast/podcast.png', help="path to source image")
    parser.add_argument("--label_file", default='./data/podcast.txt', help="path to source image")
    parser.add_argument("--border", type=int, default=0, help="Get more face for bbox")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,  help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion") 
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" ) 
    parser.add_argument("--verbose",action="store_true", help="saving the intermedia output or not" ) 
    parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" ) 

    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)


    # Additional podcast script parameters
    parser.add_argument("--slides_dir", type=str, default="data/slides", help="directory containing slides")
    parser.add_argument("--subtitles_dir", type=str, default="data/podcast/processed_audio/subtitle", help="directory containing subtitles")
    parser.add_argument("--podcast_script_path", type=str, default="data/podcast/podcast_script.json", help="path to podcast script JSON")
    parser.add_argument("--output_video_path", type=str, default="results/podcast/podcast.mp4", help="output path for the final podcast video")

    parser.add_argument("--chunk_length_ms", type=int, default=10000, help="Length of each audio chunk in milliseconds")


    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"

    save_dir = args.result_dir
    os.makedirs(save_dir, exist_ok=True)

    resize_image(args.source_image, 1782, 838)

    face_positions = read_labels(args.source_image, args.label_file, border=args.border)

    if not face_positions:
        print("No faces found.")
        return

    face_image_paths, audio_mapping = extract_faces(args.source_image, face_positions, save_dir)

    audio_path_guest = os.path.join(args.driven_audio_folder, audio_mapping["face_1.png"]) if "guest" in audio_mapping["face_1.png"] else os.path.join(args.driven_audio_folder, audio_mapping["face_2.png"])
    audio_path_host = os.path.join(args.driven_audio_folder, audio_mapping["face_2.png"]) if "host" in audio_mapping["face_2.png"] else os.path.join(args.driven_audio_folder, audio_mapping["face_1.png"])
    pic_path_guest = face_image_paths[0] if "guest" in audio_mapping["face_1.png"] else face_image_paths[1]
    pic_path_host = face_image_paths[1] if "host" in audio_mapping["face_2.png"] else face_image_paths[0]

    # Confirm with the user
    # print("Voice assigned to face_1: ", audio_mapping["face_1.png"])
    # print("Voice assigned to face_2: ", audio_mapping["face_2.png"])

    # user_input = input("Are you satisfied with these assignments? (yes/no): ")
    # user_input = "yes"
    # if user_input.lower() != "yes":
    #     # Swap face assignments
    #     audio_mapping["face_1.png"], audio_mapping["face_2.png"] = audio_mapping["face_2.png"], audio_mapping["face_1.png"]
        
    #     # Update audio paths and picture paths
    #     audio_path_guest = os.path.join(args.driven_audio_folder, audio_mapping["face_1.png"])
    #     audio_path_host = os.path.join(args.driven_audio_folder, audio_mapping["face_2.png"])
    #     pic_path_guest = face_image_paths[0]
    #     pic_path_host = face_image_paths[1]

    #     # Save the updated audio mapping
    #     with open(os.path.join(save_dir, "audio_mapping.json"), 'w') as f:
    #         json.dump(audio_mapping, f, indent=4)


    with tempfile.TemporaryDirectory() as tempdir:
        final_video_path = os.path.join(tempdir, 'final.mp4')
        process_and_append_chunks(audio_path_guest, audio_path_host, pic_path_guest, pic_path_host, face_positions, args.source_image, args, tempdir, final_video_path)

        podcast_script_command = [
            'python', 'video/podcast_video.py',
            '--video_path', final_video_path,
            '--slides_dir', args.slides_dir,
            '--subtitles_dir', args.subtitles_dir,
            '--podcast_script_path', args.podcast_script_path,
            '--output_video_path', args.output_video_path
        ]

        subprocess.run(podcast_script_command, check=True)

if __name__ == "__main__":
    main()