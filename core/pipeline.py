from pathlib import Path
from RedditVideoBot.assets.manage import load_assets_from_cfg, pick
from RedditVideoBot.audio import utils as au, fx, jumps, spatial
from RedditVideoBot.video import subtitles as submod, filters as vfx
from RedditVideoBot.analytics.store import record_render
import math

MIN_SEC=240
MAX_SEC=360

def run_pipeline(cfg:dict)->str:
    out_dir=Path(cfg.get('output_dir','outputs')); out_dir.mkdir(parents=True,exist_ok=True)
    assets=load_assets_from_cfg(cfg); mood=cfg.get('mood','any')
    bg=pick(assets['backgrounds'],mood,1)[0].path
    narr=au.load(cfg['narration']['audio'])
    text=cfg['narration']['text']
    dur_ms=len(narr)
    if dur_ms/1000 < MIN_SEC:
        narr=au.loop_to(narr, MIN_SEC*1000); dur_ms=len(narr)
    # audio fx
    afx=cfg.get('audio_fx',{})
    if afx.get('whisper'): narr=fx.whisper(narr)
    if afx.get('demon'): narr=fx.demon(narr)
    if afx.get('echo',True): narr=fx.echo(narr)
    if afx.get('reverb',True): narr=fx.reverb(narr)
    if afx.get('stereo_pan',True): narr=spatial.pan_left_right(narr)
    if (js:=cfg.get('jumpscare_sfx')):
        sfx=au.load(js['path']).apply_gain(js.get('gain_db',8))
        for pos in jumps.plan(text,dur_ms,js.get('max_events',3)):
            narr=narr.overlay(sfx,position=max(0,pos))
    narr_out=out_dir/'narr_fx.wav'; au.save(narr,str(narr_out))
    # bg+subs
    dur=dur_ms/1000
    bg_fx=out_dir/'bg.mp4'; vfx.apply_overlays(bg,str(bg_fx),dur)
    srt_path=out_dir/'subs.srt'; srt_path.write_text(submod.make_srt(text,dur),encoding='utf-8')
    subbed=out_dir/'subbed.mp4'; vfx.burn_subs(str(bg_fx),str(srt_path),str(subbed))
    final=out_dir/(cfg.get('output_name','out.mp4')); vfx.mux(str(subbed),str(narr_out),str(final))
    # enforce split if too long
    if dur>=MAX_SEC:
        parts=[]; subs=submod.split_srt(srt_path.read_text(),MAX_SEC)
        for i,seg in enumerate(subs,1):
            pf=out_dir/f"part{i}.mp4"
            vfx.cut_video(str(final),pf,seg['start'],seg['end'])
            parts.append(str(pf))
        out=parts
    else:
        out=[str(final)]
    record_render(str(out_dir),cfg,";".join(out),dur_ms,"ok")
    return out[0]
