from pydub import AudioSegment, effects
from pydub.generators import WhiteNoise, Sine

def whisperize(seg: AudioSegment) -> AudioSegment:
    noise = WhiteNoise().to_audio_segment(duration=len(seg)).-20
    mixed = seg.low_pass_filter(3000).overlay(noise, gain_during_overlay=-6)
    return effects.normalize(mixed)

def demonic(seg: AudioSegment) -> AudioSegment:
    oct_down = seg._spawn(seg.raw_data, overrides={"frame_rate": int(seg.frame_rate*0.7)}).set_frame_rate(seg.frame_rate)
    return effects.normalize(oct_down)

def echo(seg: AudioSegment, delay_ms:int=180, decay:float=0.45) -> AudioSegment:
    echo = AudioSegment.silent(duration=len(seg)+delay_ms)
    echo = echo.overlay(seg - 6, position=delay_ms)
    combined = seg.overlay(echo - int(6*(1-decay)))
    return effects.normalize(combined)

def reverb_fake(seg: AudioSegment, taps:int=3, base_delay:int=70) -> AudioSegment:
    out = seg
    for i in range(1,taps+1):
        out = out.overlay(seg - (6*i), position=base_delay*i)
    return effects.normalize(out)
