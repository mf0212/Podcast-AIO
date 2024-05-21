import json
import os
#import openai
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import textwrap
import argparse

# def generate_question(key_question):
#     openai.api_key = "sk-proj-TyKHo0iWch01g1Ouuw1NT3BlbkFJtb4KVpGfoP1FiEB7vkxm"

#     prompt = f"""
#     Extract ONLY the core question from the following script snippet, removing any greetings, unnecessary phrases, or personal pronouns. The question should focus on a specific aspect of Video Classification and be as concise as possible. DO NOT add any extra words or phrases like "Câu hỏi của bạn:, Câu hỏi:, Your Question:, Core Question: ...:" in output:

#     Original Script:
#     {key_question}

#     Examples of desired output:
#     - Original: Chào mừng bạn đến với chương trình của chúng ta hôm nay! Rất vui được trò chuyện với một chuyên gia về AI như bạn. Trước tiên, bạn có thể cho chúng ta biết thêm về Video Classification là gì không?
#         Your Question: Video Classification là gì?

#     - Original: Nghe hấp dẫn đấy! Vậy dữ liệu video được xử lý như thế nào trong bài toán này?
#         Your Question: Dữ liệu video được xử lý như thế nào?

#     - Original: Bạn có thể giải thích thêm về VideoMAE và cách thức hoạt động của nó không?
#         Your Question: VideoMAE hoạt động như thế nào?
#     - Original: "Cảm ơn bạn đã chia sẻ những kiến thức bổ ích về Video Classification. Bạn có muốn chia sẻ thêm điều gì với khán giả của chúng ta không?"
#         Your Question: Những lưu ý về Video Classification ?

#     Your Question:
#     """

#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are an assistant helping to extract concise and impersonal questions from podcast scripts in Vietnamese."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.15,
#         top_p=0.6,
#         presence_penalty=0.8,
#     )

#     question = response['choices'][0]['message']['content'].strip()
#     #Post-processing: Split by first colon and take the second part
#     parts = question.split(':', 1)
#     if len(parts) > 1:
#         question = parts[1].strip()
#     return question

# Function to generate a subtitle frame
def generate_subtitle_frame(text, title, video_size):
    # Create a blank image with the same height as the video
    img = Image.new('RGBA', (video_size[0], video_size[1]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Load a font
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Adjust the path as necessary
    font = ImageFont.truetype(font_path, 16)
    title_font = ImageFont.truetype(font_path, 40)

    
    # Generate question from key_question
    # question = generate_question(key_question)

    # Define the area for the question (blue box at the top)
    title_box_y = 10  # Adjust y-position as needed
    title_box_height = 100  # Adjust height as needed
    title_box_width = video_size[0] - 200  # Leave margins on sides

    # question_box_y = 25  # Adjust y-position as needed
    # question_box_height = 80  # Adjust height as needed
    # question_box_width = video_size[0] - 200  # Leave margins on sides

    # Define the area where the subtitles should be drawn (blue box area)
    sub_box_y = 747  # Adjust this value based on the y-position of the sub box
    sub_box_height = 100  # Adjust this value based on the height of the sub box
    sub_box_width = video_size[0] - 200  # Adjust this value based on the width of the sub box (leave margin on sides)
    
    # Wrap the text to fit within the sub box width
    wrapped_text = textwrap.fill(text, width=100)  # Adjust the width value as necessary
    #wrapped_question = textwrap.fill(question, width=100)
    wrapped_title = textwrap.fill(title, width=100)

    # Split the wrapped text into lines
    lines = wrapped_text.split('\n')
    #question_lines = wrapped_question.split("\n")
    title_lines = wrapped_title.split("\n")

    # Calculate the height of the text block
    text_height = sum([draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines])
    #question_height = sum([draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in question_lines])
    title_height = sum([draw.textbbox((0, 0), line, font=title_font)[3] - draw.textbbox((0, 0), line, font=title_font)[1] for line in title_lines])
    

    # Calculate the y position to center the text block within the sub box
    text_y = sub_box_y + (sub_box_height - text_height) // 2
    #question_y = question_box_y + (question_box_height - question_height) // 2
    title_y = title_box_y + (title_box_height - title_height) // 2
    
    
    for line in title_lines:
        line_width = draw.textbbox((0, 0), line, font=title_font)[2] - draw.textbbox((0, 0), line, font=title_font)[0]
        title_x = (video_size[0] - line_width) // 2  # Center each line horizontally
        draw.text((title_x, title_y), line, font=title_font, fill="black")
        title_y += draw.textbbox((0, 0), line, font=title_font)[3] - draw.textbbox((0, 0), line, font=title_font)[1]

    # Draw each line of text onto the image
    for line in lines:
        line_width = draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0]
        text_x = (video_size[0] - line_width) // 2  # Center each line horizontally
        draw.text((text_x, text_y), line, font=font, fill="black")
        text_y += draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]

    # for line in question_lines:
    #     line_width = draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0]
    #     question_x = (video_size[0] - line_width) // 2  # Center each line horizontally
    #     draw.text((question_x, question_y), line, font=font, fill="black")
    #     question_y += draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]


    return np.array(img)

def main(video_path, slides_dir, subtitles_dir, podcast_script_path, output_video_path):
    # Load the video
    video = VideoFileClip(video_path)
    video_size = video.size

    # Check if the fps attribute is set correctly
    if not hasattr(video, 'fps') or video.fps is None:
        video.fps = 25  # Set a default FPS value if it's not set
    print(f"Video FPS: {video.fps}")

    # Load podcast script to get the correct order
    with open(podcast_script_path, 'r', encoding='utf-8') as f:
        podcast_data = json.load(f)

    # Create a list to hold all clips
    clips = [video]

    # Iterate through each slide in the order specified in the podcast script
    current_time = 0
    for entry in podcast_data:
        slide_number = entry['slide']
        slide_path = os.path.join(slides_dir, f"{slide_number}.jpg")
        json_path = os.path.join(subtitles_dir, f"{slide_number}.json")
        
        if os.path.exists(slide_path):
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    subtitles_data = json.load(f)
                slide_duration = sum(item['duration'] for item in subtitles_data)
                
                desired_slide_width = 970
                desired_slide_height = 490
                
                slide_clip = ImageClip(slide_path).set_duration(slide_duration).set_start(current_time).resize(newsize=(desired_slide_width, desired_slide_height)).set_position(('center', 'center'))
                clips.append(slide_clip)

                print(f"Slide {slide_number}: Duration = {slide_duration}, Start time = {current_time}")


                # Generate subtitle clips for each subtitle in the JSON
                for subtitle_data in subtitles_data:
                    text = subtitle_data['subtitle']
                    duration = subtitle_data['duration']
                    #key_question = entry.get('question', "")
                    title = "CHÀO MỪNG BẠN ĐẾN VỚI PODCAST AI CỦA AIVN"
                    # Generate a subtitle frame
                    subtitle_frame = generate_subtitle_frame(text, title, video_size)  # Pass key_question
                    
                    # Convert the frame to an ImageClip
                    subtitle_clip = ImageClip(subtitle_frame).set_duration(duration).set_start(current_time).set_position(('center', 'bottom'))
                    
                    clips.append(subtitle_clip)
                    current_time += duration
                

    # Print durations of all clips
    for idx, clip in enumerate(clips):
        print(f"Clip {idx} duration: {clip.duration}")

    # Check for clips with invalid durations
    invalid_clips = [idx for idx, clip in enumerate(clips) if clip.duration is None]
    if invalid_clips:
        print(f"Error: The following clips have invalid durations: {invalid_clips}")
        raise ValueError(f"Clips with invalid durations: {invalid_clips}")

    # Combine all clips into one video
    final_clip = CompositeVideoClip(clips, size=video_size)

   #sub_final_clip = final_clip.subclip(0, 5)


    # Write the final video to file
    print("Attempting to write the final video to file...")
    final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', fps=video.fps)
    #sub_final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', fps=video.fps)

    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a podcast video with slides and subtitles.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--slides_dir', type=str, required=True, help='Directory containing slide images.')
    parser.add_argument('--subtitles_dir', type=str, required=True, help='Directory containing JSON subtitle files.')
    parser.add_argument('--podcast_script_path', type=str, required=True, help='Path to the podcast script JSON file.')
    parser.add_argument('--output_video_path', type=str, required=True, help='Path to the output video file.')

    args = parser.parse_args()
    main(args.video_path, args.slides_dir, args.subtitles_dir, args.podcast_script_path, args.output_video_path)

#python podcast_video.py --video_path test.mp4 --slides_dir data/slides --subtitles_dir data/podcast/processed_audio/subtitle --podcast_script_path data/podcast/podcast_script.json --output_video_path podcast.mp4
