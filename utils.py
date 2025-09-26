
import subprocess
from moviepy.editor import VideoFileClip
import os

def run_ff(cmd:list):
    subprocess.run(cmd, check=True)

def split_video(video_path, max_duration=720):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    output_files = []
    if duration <= max_duration:
        clip.close()
        return [video_path]
    base, ext = os.path.splitext(video_path)
    start = 0
    part = 1
    while start < duration:
        end = min(start + max_duration, duration)
        subclip = clip.subclip(start, end)
        out_file = f"{base}_part{part}{ext}"
        subclip.write_videofile(out_file, codec="libx264", audio_codec="aac")
        output_files.append(out_file)
        start += max_duration
        part += 1
    clip.close()
    return output_files
