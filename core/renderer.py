import ffmpeg
from pathlib import Path

def burn_subtitles(video_path:str, srt_path:str, out_path:str, font_size:int=24):
    # Requires ffmpeg with libass
    (
        ffmpeg
        .input(video_path)
        .filter('subtitles', srt_path, force_style=f'Fontsize={font_size}')
        .output(out_path, crf=18, preset='medium', vcodec='libx264', acodec='aac', movflags='+faststart')
        .overwrite_output()
        .run(quiet=True)
    )
    return out_path

def concat_bg_with_music(bg_path:str, music_path:str, out_path:str, duration:float):
    # Trim/loop is expected to be preprocessed to exact duration lengths
    v = ffmpeg.input(bg_path, stream_loop=-1, t=duration)
    a = ffmpeg.input(music_path, stream_loop=-1, t=duration)
    (
        ffmpeg
        .output(v, a, out_path, shortest=None, crf=18, preset='medium', vcodec='libx264', acodec='aac', movflags='+faststart')
        .overwrite_output()
        .run(quiet=True)
    )
    return out_path

def overlay_watermark(video_path:str, watermark_png:str, out_path:str, x:str='(w-w*0.2)-20', y:str='20'):
    (
        ffmpeg
        .input(video_path)
        .overlay(ffmpeg.input(watermark_png), x=x, y=y, eof_action='pass')
        .output(out_path, crf=18, preset='medium', vcodec='libx264', acodec='copy', movflags='+faststart')
        .overwrite_output()
        .run(quiet=True)
    )
    return out_path
