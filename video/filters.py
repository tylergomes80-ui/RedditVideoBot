import ffmpeg
def apply_overlays(bg_path:str,out_path:str,duration:float):
    v=ffmpeg.input(bg_path,stream_loop=-1,t=duration)
    v=v.filter('scale','-2','720')
    ffmpeg.output(v,out_path,vcodec='libx264',crf=18).overwrite_output().run(quiet=True)
def burn_subs(video_path,srt_path,out_path):
    ffmpeg.input(video_path).filter('subtitles',srt_path).output(out_path,vcodec='libx264',crf=18).overwrite_output().run(quiet=True)
def mux(v,a,o): ffmpeg.output(ffmpeg.input(v),ffmpeg.input(a),o,vcodec='copy',acodec='aac').overwrite_output().run(quiet=True)
def cut_video(inp,out,start,end):
    ffmpeg.input(inp,ss=start,to=end).output(out,vcodec='libx264',acodec='copy').overwrite_output().run(quiet=True)
