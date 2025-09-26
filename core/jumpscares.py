from pydub import AudioSegment
from typing import List, Tuple

KEYWORDS = {"scream","blood","shadow","behind","door","run","dead"}

def plan_from_text(text:str, duration_ms:int, n:int=3)->List[int]:
    words = text.lower().split()
    hits = [i for i,w in enumerate(words) if w.strip(',.!?') in KEYWORDS]
    if not hits:
        step = max(1, duration_ms//(n+1))
        return [step*(i+1) for i in range(n)]
    # map word index to time
    ms_per_word = max(200, duration_ms//max(1,len(words)))
    return [min(duration_ms-1, i*ms_per_word) for i in hits[:n]]

def apply_jumps(seg: AudioSegment, sfx: AudioSegment, positions_ms: List[int], gain:int=6)->AudioSegment:
    out = seg
    for pos in positions_ms:
        out = out.overlay(sfx + gain, position=max(0,pos))
    return out
