from typing import List, Tuple
import srt
from datetime import timedelta
import math
import re

def segment_text(text:str, words_per_line:int=8) -> List[str]:
    words = text.split()
    chunks = [" ".join(words[i:i+words_per_line]) for i in range(0,len(words),words_per_line)]
    return chunks

def estimate_timing(total_seconds:float, n_chunks:int)->List[Tuple[float,float]]:
    dur = total_seconds/max(1,n_chunks)
    out=[]; t=0.0
    for i in range(n_chunks):
        start=t; end=min(total_seconds, t+dur)
        out.append((start,end))
        t=end
    return out

def make_srt(text:str, total_seconds:float)->str:
    parts = segment_text(text)
    times = estimate_timing(total_seconds, len(parts))
    subs=[]
    for i,(chunk,(start,end)) in enumerate(zip(parts,times), start=1):
        subs.append(srt.Subtitle(index=i, start=timedelta(seconds=start), end=timedelta(seconds=end), content=chunk))
    return srt.compose(subs)
