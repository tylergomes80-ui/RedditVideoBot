import srt
from datetime import timedelta
def make_srt(text:str,total:float)->str:
    words=text.split(); chunk=8
    parts=[" ".join(words[i:i+chunk]) for i in range(0,len(words),chunk)]
    dur=total/max(1,len(parts)); out=[]; t=0.0
    for i,p in enumerate(parts,1):
        out.append(srt.Subtitle(index=i,start=timedelta(seconds=t),end=timedelta(seconds=min(total,t+dur)),content=p))
        t+=dur
    return srt.compose(out)

def split_srt(srt_text:str,max_sec:int):
    subs=list(srt.parse(srt_text)); parts=[]; start=0; buf=[]
    for s in subs:
        sec=s.end.total_seconds()
        if sec-start>max_sec:
            parts.append({"start":start,"end":sec,"subs":buf}); buf=[]; start=sec
        buf.append(s)
    if buf: parts.append({"start":start,"end":subs[-1].end.total_seconds(),"subs":buf})
    return parts
