from pydub import AudioSegment

def load_audio(path:str) -> AudioSegment:
    return AudioSegment.from_file(path)

def save_audio(seg: AudioSegment, path:str):
    seg.export(path, format=path.split(".")[-1])

def loop_to_duration(seg: AudioSegment, ms:int) -> AudioSegment:
    if len(seg)>=ms:
        return seg[:ms]
    out = AudioSegment.silent(duration=0)
    while len(out)<ms:
        out += seg
    return out[:ms]
