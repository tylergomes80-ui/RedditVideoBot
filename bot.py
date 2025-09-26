# bot.py â€” RedditVideoBot single-file build (clean, Azure-only, debug-enabled)

# Features:
# - Config loader + validation
# - Asset Manager: HTTP + YouTube (yt_dlp py or CLI), dedupe, resume
# - Reddit fetch: PRAW or public JSON fallback, score/length filters, debug logs, fail-fast if no posts
# - Planner: sentence-aware chunking, ~160w shots, durable shotlist.json
# - TTS: Azure Speech only (no edge-tts, no pyttsx3). Fails if Azure not configured.
# - Subtitles: SRT + ASS export; burn-in via PILâ†’ImageClip (no ImageMagick)
# - Renderer: 9:16 vertical, smart crop/scale, looped backgrounds
# - Audio: narration + optional music with simple ducking, loudness normalize
# - Outputs: long 1080x1920 and 720x1280; optional shorts by max duration
# - Tools: split_video, backup, clean, full pipeline
# - Menus: granular steps + full run

# Dependencies (pip): pyyaml, requests, rich, moviepy, numpy, pillow, azure-cognitiveservices-speech
# Optional: praw, yt-dlp, ffmpeg in PATH

from __future__ import annotations

import os
import sys
import re
import json
import math
import yaml
import time
import zipfile
import shutil
import random
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

# ---- Optional Azure SDK (required for TTS in this build) ----
try:
    import azure.cognitiveservices.speech as speechsdk  # type: ignore
except Exception:
    speechsdk = None  # will fail-fast if used without install

# ---- Optional libs ----
try:
    import praw  # Reddit API
except Exception:
    praw = None

try:
    import yt_dlp  # YouTube downloader (Python)
except Exception:
    yt_dlp = None

import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips,
    ImageClip, ColorClip
)
# Correct function imports from moviepy (avoid module-is-not-callable)
from moviepy.audio.fx.all import audio_normalize, volumex
from moviepy.video.fx.all import crop as vfx_crop, resize as vfx_resize

console = Console()
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
OUTPUT = ROOT / "output"
AUDIO = ROOT / "audio"
DATA = ROOT / "data"
CONFIG_FILE = ROOT / "config.yaml"

# Defaults
V_HEIGHT = 1920
V_WIDTH = 1080   # 9:16 vertical
V_WIDTH_SMALL = 720
FPS = 30

# -----------------------------
# Utilities
# -----------------------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def slugify(txt: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^\w\-]+", "_", txt.strip())
    s = re.sub(r"__+", "_", s)
    return s[:maxlen].strip("_").lower() or "file"

def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url

def ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

def load_config() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        console.print("[red]Missing config.yaml[/red]")
        sys.exit(1)
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # ---- Reddit defaults ----
    cfg.setdefault("reddit", {})
    cfg["reddit"].setdefault("limit", 5)
    cfg["reddit"].setdefault("min_score", 300)   # looser default to get results
    cfg["reddit"].setdefault("max_length", 3000) # looser default
    cfg["reddit"].setdefault("subreddit", "AskReddit")

    # ---- Azure-only TTS ----
    cfg.setdefault("azure", {})
    cfg["azure"].setdefault("key", None)
    cfg["azure"].setdefault("region", None)
    cfg["azure"].setdefault("voice", "en-US-AriaNeural")
    cfg.setdefault("tts", {})
    cfg["tts"]["engine"] = "azure"
    cfg["tts"].setdefault("voice", cfg["azure"].get("voice", "en-US-AriaNeural"))
    cfg["tts"].setdefault("rate", "+0%")
    cfg["tts"].setdefault("pitch", "+0Hz")
    cfg["tts"].setdefault("output", str((AUDIO / "narration.wav").resolve()))

    # ---- Render ----
    cfg.setdefault("render", {})
    # Font: try config value first. If absent, try common Windows fonts. Else DejaVuSans by name. Else PIL default.
    cfg["render"].setdefault("font", r"C:\Windows\Fonts\arial.ttf")
    cfg["render"].setdefault("font_size", 52)
    cfg["render"].setdefault("line_height", 1.25)
    cfg["render"].setdefault("subtitle_box", True)
    cfg["render"].setdefault("subtitle_max_width_px", int(V_WIDTH * 0.86))
    cfg["render"].setdefault("subtitle_bottom_margin_px", 180)
    cfg["render"].setdefault("bg_blur", False)
    cfg["render"].setdefault("music_volume", 0.08)  # 8%
    cfg["render"].setdefault("music_duck", 0.6)     # music multiplier under narration
    cfg["render"].setdefault("shorts_max_seconds", 60)
    cfg["render"].setdefault("make_shorts", True)
    cfg["render"].setdefault("vertical_resolution", [V_WIDTH, V_HEIGHT])  # [w,h]

    cfg.setdefault("assets", {})
    cfg["assets"].setdefault("backgrounds", [])
    cfg["assets"].setdefault("music", [])
    cfg["assets"].setdefault("sfx", [])
    return cfg

config = load_config()
safe_mkdir(ASSETS); safe_mkdir(OUTPUT); safe_mkdir(AUDIO); safe_mkdir(DATA)

# -----------------------------
# Asset Manager
# -----------------------------
def _download_http(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r.raw, f)

def _download_youtube(url: str, dest_dir: Path) -> Optional[Path]:
    """
    Download best video or audio depending on target directory.
    If dest_dir name is 'music' or 'sfx', prefer audio-only.
    """
    is_audio_cat = dest_dir.name in ("music", "sfx")
    ext = "m4a" if is_audio_cat else "mp4"
    outtmpl = str(dest_dir / "%(title).80s.%(ext)s")

    if yt_dlp is not None:
        ydl_opts = {
            "outtmpl": outtmpl,
            "noprogress": True,
            "quiet": True,
            "merge_output_format": ext,
        }
        if is_audio_cat:
            ydl_opts["format"] = "bestaudio/best"
            ydl_opts["postprocessors"] = [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
                "preferredquality": "5",
            }]
        else:
            ydl_opts["format"] = "bv*+ba/b[ext=mp4]/b"
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if "requested_downloads" in info and info["requested_downloads"]:
                    p = Path(info["requested_downloads"][0]["filepath"])
                    return p
                if "title" in info:
                    # Fallback: guess
                    for f in dest_dir.glob(f"{info['title']}*.{ext}"):
                        return f
        except Exception as e:
            console.print(f"[yellow]yt_dlp python failed[/yellow]: {e}")

    # fallback CLI
    try:
        args = ["yt-dlp", "-o", outtmpl, "-q"]
        if is_audio_cat:
            args += ["-f", "bestaudio/best", "--extract-audio", "--audio-format", "m4a"]
        else:
            args += ["-f", "bv*+ba/b[ext=mp4]/b", "--merge-output-format", "mp4"]
        subprocess.run(args + [url], check=True)
        files = sorted(dest_dir.glob(f"*.{ext}"), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0] if files else None
    except Exception as e:
        console.print(f"[red]yt-dlp CLI failed[/red]: {e}")
        return None

def download_assets(category: str) -> None:
    urls = config.get("assets", {}).get(category, [])
    if not urls:
        console.print(f"[yellow]No URLs configured for {category}[/yellow]")
        return

    catdir = ASSETS / category
    safe_mkdir(catdir)

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(), TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Downloading {category}", total=len(urls))

        for url in urls:
            try:
                if is_youtube(url):
                    out = _download_youtube(url, catdir)
                    if out and out.exists():
                        console.print(f"[green]YouTube ok[/green] {out.name}")
                    else:
                        console.print(f"[red]YouTube failed[/red] {url}")
                else:
                    fname = url.split("?")[0].split("/")[-1] or f"{slugify(url)}"
                    dest = catdir / fname
                    if dest.exists():
                        console.print(f"[yellow]Skip existing[/yellow] {dest.name}")
                    else:
                        _download_http(url, dest)
                        console.print(f"[green]HTTP ok[/green] {dest.name}")
            except Exception as e:
                console.print(f"[red]Failed[/red] {url}: {e}")
            progress.advance(task)

# -----------------------------
# Reddit Fetch
# -----------------------------
def _reddit_public_fetch(subreddit: str, limit: int) -> List[Dict[str, Any]]:
    # Simple public JSON scrape of /r/{sub}/hot.json
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={min(limit*5,100)}"
    headers = {"User-Agent": "RedditVideoBot/1.0"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    items: List[Dict[str, Any]] = []
    for child in data.get("data", {}).get("children", []):
        console.print("[cyan]DEBUG[/cyan] Public fetch child found")
        p = child.get("data", {}) or {}
        body = p.get("selftext") or ""
        if not body:
            continue
        items.append({
            "title": p.get("title", ""),
            "body": body,
            "score": int(p.get("score", 0) or 0),
            "id": p.get("id", ""),
            "permalink": "https://reddit.com" + p.get("permalink", ""),
        })
    return items

def fetch_reddit_stories():
    subs = config["reddit"].get("subreddit", "AskReddit")
    if isinstance(subs, str):
        subs = [subs]
    subs = subs[:]  # copy
    random.shuffle(subs)
    console.print(f"[cyan]DEBUG[/cyan] Subreddit candidates: {subs}")

    limit = int(config["reddit"].get("limit", 5))
    min_score = int(config["reddit"].get("min_score", 1000))
    max_len = int(config["reddit"].get("max_length", 1200))

    posts: List[Dict[str, Any]] = []

    for sub in subs:
        console.print(f"[cyan]DEBUG[/cyan] Trying subreddit: {sub}")

        # First try PRAW
        if praw and all(k in config["reddit"] for k in ("client_id", "client_secret")):
            try:
                reddit = praw.Reddit(
                    client_id=config["reddit"]["client_id"],
                    client_secret=config["reddit"]["client_secret"],
                    user_agent=config["reddit"]["user_agent"],
                )
                for post in reddit.subreddit(sub).hot(limit=limit * 5):
                    body = getattr(post, "selftext", "") or ""
                    console.print(f"[cyan]DEBUG[/cyan] Considering {post.id} score={post.score} len={len(body)}")
                    if post.score < min_score:
                        console.print(f"Reject {post.id}: score {post.score} < {min_score}")
                        continue
                    if len(body) < 200 or len(body) > max_len:
                        console.print(f"Reject {post.id}: len {len(body)} outside [200,{max_len}]")
                        continue
                    posts.append({
                        "title": post.title,
                        "body": body,
                        "score": int(post.score),
                        "id": post.id,
                        "permalink": f"https://reddit.com{post.permalink}"
                    })
                    if len(posts) >= limit:
                        break
            except Exception as e:
                console.print(f"[yellow]PRAW failed for {sub}[/yellow]: {e}")

        # If nothing from PRAW, try public JSON
        if not posts:
            try:
                pub = _reddit_public_fetch(sub, limit)
                for p in pub:
                    console.print("[cyan]DEBUG[/cyan] Public fetch child found")
                posts = [
                    p for p in pub
                    if p["score"] >= min_score and 200 <= len(p["body"]) <= max_len
                ][:limit]
            except Exception as e:
                console.print(f"[red]Public fetch failed for {sub}[/red]: {e}")

        if posts:
            console.print(f"[green]Fetched {len(posts)} stories from r/{sub}[/green]")
            break

    if not posts:
        raise RuntimeError("No valid Reddit posts fetched. Loosen filters or try different subreddits.")

    DATA.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "stories_meta.json", "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2)

    (ROOT / "story.txt").write_text(posts[0]["title"] + "\n\n" + posts[0]["body"], encoding="utf-8")


# -----------------------------
# Planner
# -----------------------------
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z0-9\"'])")

def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    parts = SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]

def _split_for_length(sentences: List[str], max_words: int = 160) -> List[str]:
    chunks, cur, wc = [], [], 0
    for s in sentences:
        sw = s.split()
        if wc + len(sw) > max_words and cur:
            chunks.append(" ".join(cur))
            cur, wc = [], 0
        cur.append(s)
        wc += len(sw)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def generate_shotlist() -> None:
    story_file = ROOT / "story.txt"
    if not story_file.exists():
        console.print("[red]story.txt missing[/red]")
        return
    raw = story_file.read_text(encoding="utf-8").strip()
    if "\n\n" in raw:
        title, body = raw.split("\n\n", 1)
    else:
        title, body = "Reddit Story", raw

    sentences = _split_sentences(body)
    chunks = _split_for_length(sentences, max_words=160)

    bg_files: List[Path] = []
    for ext in ("*.mp4","*.mov","*.mkv","*.webm"):
        bg_files += list((ASSETS / "backgrounds").glob(ext))

    if not bg_files:
        console.print("[yellow]No backgrounds found. Will use solid color.[/yellow]")

    # Estimate duration per chunk at ~2.6 wps
    shots = []
    for i, chunk in enumerate(chunks):
        words = chunk.split()
        est = max(4, int(len(words) / 2.6))
        bg = str(random.choice(bg_files)) if bg_files else None
        shots.append({
            "idx": i,
            "text": chunk,
            "background": bg,
            "est_duration": est
        })

    with open(ROOT / "shotlist.json", "w", encoding="utf-8") as f:
        json.dump({
            "title": title,
            "shots": shots
        }, f, indent=2)
    console.print(f"[green]Shotlist with {len(shots)} shots created[/green]")

# -----------------------------
# TTS (Azure only) + Subtitle Rendering
# -----------------------------
def tts_generate():
    """
    Generate narration with Azure TTS and save word timings for subtitles.
    Produces audio/output.wav and audio/word_timings.json
    """
    from azure.cognitiveservices.speech import (
        SpeechConfig, SpeechSynthesizer, AudioConfig, ResultReason, SpeechSynthesisOutputFormat
    )
    import wave, io, json

    out_wav = Path("audio/output.wav")
    timings_path = Path("audio/word_timings.json")
    script_path = Path("story.txt")

    if not script_path.exists():
        raise RuntimeError("story.txt not found. Run fetch_reddit_stories and shotlist first.")

    text = script_path.read_text(encoding="utf-8").replace("\n", " ").strip()
    voice = config["azure"]["voice"]
    key = config["azure"]["key"]
    region = config["azure"]["region"]

    speech_config = SpeechConfig(subscription=key, region=region)
    speech_config.speech_synthesis_voice_name = voice
    speech_config.set_speech_synthesis_output_format(
        SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
    )

    synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    # Azure safe limit ~3000 chars per request
    max_chars = 2800
    chunks = []
    while len(text) > 0:
        chunk = text[:max_chars]
        cut = max(chunk.rfind("."), chunk.rfind("?"), chunk.rfind("!"))
        if cut > 0 and cut > max_chars // 2:
            chunk, text = text[:cut+1], text[cut+1:].lstrip()
        else:
            chunk, text = chunk, text[len(chunk):].lstrip()
        chunks.append(chunk)

    all_timings = []

    with wave.open(str(out_wav), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(16000)

        current_offset = 0.0

        for idx, chunk in enumerate(chunks):
            timings = []

            def on_word_boundary(evt):
                timings.append({
                    "word": evt.text,
                    "offset": int(evt.audio_offset),   # 100ns ticks
                    "duration": int(evt.duration),     # 100ns ticks
                })

            synthesizer.synthesis_word_boundary.connect(on_word_boundary)

            result = synthesizer.speak_text_async(chunk).get()
            if result.reason != ResultReason.SynthesizingAudioCompleted:
                raise RuntimeError(f"Azure TTS failed on chunk {idx}: {result.reason}")

            # Write audio
            audio_data = result.audio_data
            buf = io.BytesIO(audio_data)
            with wave.open(buf, "rb") as wav_in:
                frames = wav_in.readframes(wav_in.getnframes())
                wav_out.writeframes(frames)

            # Adjust timings to absolute seconds
            for t in timings:
                start_sec = current_offset + (t["offset"] / 10_000_000)
                all_timings.append({
                    "word": t["word"],
                    "offset": int(start_sec * 10_000_000),  # back to 100ns
                    "duration": t["duration"]
                })

            # Update offset by chunk duration
            current_offset = wav_out.getnframes() / 16000.0

            synthesizer.synthesis_word_boundary.disconnect_all()

    with open(timings_path, "w", encoding="utf-8") as f:
        json.dump(all_timings, f, indent=2)

    console.print(f"[green]Azure TTS narration saved to {out_wav}, timings saved to {timings_path}[/green]")


# -----------------------------
# Subtitle Rendering from Azure timings
# -----------------------------
def build_subtitles_from_timings(timings_path: Path, render_cfg: dict):
    """
    Build MoviePy subtitle clips from Azure word timings.
    Groups words into sentences, syncs start/end times, and applies fade effects.
    """
    import json
    from moviepy.editor import TextClip
    from moviepy.video.fx.all import fadein, fadeout

    with open(timings_path, "r", encoding="utf-8") as f:
        words = json.load(f)

    # Group into sentences
    sentences = []
    current = {"words": [], "start": None, "end": None}
    for w in words:
        text = w["word"]
        start = w["offset"] / 10_000_000  # 100ns → seconds
        dur = w["duration"] / 10_000_000
        end = start + dur

        if current["start"] is None:
            current["start"] = start
        current["words"].append(text)
        current["end"] = end

        if any(text.endswith(p) for p in [".", "?", "!"]):
            sentences.append({
                "text": " ".join(current["words"]).strip(),
                "start": current["start"],
                "end": current["end"],
            })
            current = {"words": [], "start": None, "end": None}

    if current["words"]:
        sentences.append({
            "text": " ".join(current["words"]).strip(),
            "start": current["start"],
            "end": current["end"],
        })

    clips = []
    for s in sentences:
        txt = TextClip(
            s["text"],
            fontsize=render_cfg.get("font_size", 40),
            font=render_cfg.get("font", "Arial"),
            color=render_cfg.get("text_color", "white"),
            stroke_color=render_cfg.get("stroke_color", "black"),
            stroke_width=render_cfg.get("stroke_width", 2),
            method="caption",
            size=(1080, None),
            align="center",
        ).set_position(("center", "bottom"))

        txt = txt.set_start(s["start"]).set_duration(s["end"] - s["start"])
        txt = fadein(txt, 0.2)
        txt = fadeout(txt, 0.2)
        clips.append(txt)

    return clips

# -----------------------------
# Subtitles: SRT/ASS + burn-in
# -----------------------------
def _srt_escape(txt: str) -> str:
    return txt.replace("\n", " ").strip()

def _format_ts(t: float) -> str:
    h = int(t // 3600); t -= h*3600
    m = int(t // 60); t -= m*60
    s = int(t); ms = int((t - s)*1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def build_word_mapping(shots: List[Dict[str, Any]], word_timings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # If timings empty, fallback to estimates.
    mapping = []
    if not word_timings:
        t = 0.0
        for s in shots:
            dur = float(s.get("est_duration", 0)) or 0.5
            mapping.append({"idx": s["idx"], "start": t, "end": t+dur, "text": s["text"]})
            t += dur
        return mapping

    # Timings present (not used in current Azure-simple path)
    words_all = [(wt["w"], wt["t"], wt.get("d", 0.0)) for wt in word_timings]
    wpos = 0
    for s in shots:
        shot_words = s["text"].split()
        if not shot_words:
            mapping.append({"idx": s["idx"], "start": words_all[wpos][1] if wpos < len(words_all) else 0.0,
                            "end": words_all[wpos][1] if wpos < len(words_all) else 0.0, "text": ""})
            continue
        start_t = words_all[wpos][1] if wpos < len(words_all) else 0.0
        wpos += len(shot_words) - 1
        end_t = (words_all[wpos][1] + words_all[wpos][2]) if wpos < len(words_all) else start_t + float(s.get("est_duration", 0))
        wpos += 1
        mapping.append({"idx": s["idx"], "start": start_t, "end": end_t, "text": s["text"]})
    # Ensure monotonic
    last = 0.0
    for m in mapping:
        if m["start"] < last:
            m["start"] = last
        if m["end"] <= m["start"]:
            m["end"] = m["start"] + 0.5
        last = m["end"]
    return mapping

def write_srt(mapping: List[Dict[str, Any]], out_path: Path) -> None:
    lines = []
    for i, m in enumerate(mapping, 1):
        a = _format_ts(m["start"]); b = _format_ts(m["end"])
        lines.append(str(i)); lines.append(f"{a} --> {b}"); lines.append(_srt_escape(m["text"])); lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")

def write_ass(mapping: List[Dict[str, Any]], out_path: Path, res: Tuple[int,int] = (V_WIDTH, V_HEIGHT)) -> None:
    w, h = res
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {w}",
        f"PlayResY: {h}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        ("Style: Default,DejaVu Sans,46,&H00FFFFFF,&H000000FF,&H00111111,&H80000000,"
         "0,0,0,0,100,100,0,0,1,2,0,2,60,60,80,1"),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    def _ass_ts(t: float) -> str:
        h = int(t // 3600); t -= h*3600
        m2 = int(t // 60); t -= m2*60
        s = int(t); cs = int((t - s)*100)
        return f"{h:d}:{m2:02d}:{s:02d}.{cs:02d}"
    body = [f"Dialogue: 0,{_ass_ts(m['start'])},{_ass_ts(m['end'])},Default,,0,0,0,,{m['text'].replace(',', 'ï¼Œ')}" for m in mapping]
    out_path.write_text("\n".join(header + body), encoding="utf-8")

# -----------------------------
# Subtitle Burn-in (PIL â†’ ImageClip)
# -----------------------------
def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    # Robust multi-fallback with logging
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates += [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        "DejaVuSans.ttf",
    ]
    for fp in candidates:
        try:
            font = ImageFont.truetype(fp, size=size)
            console.print(f"[cyan]Font loaded[/cyan] {fp}")
            return font
        except Exception:
            continue
    console.print("[yellow]All TTF attempts failed. Using PIL default font[/yellow]")
    return ImageFont.load_default()

def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> List[str]:
    words = text.split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        tw, th = draw.textbbox((0,0), test, font=font)[2:]
        if tw <= max_w or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur)); cur = [w]
    if cur: lines.append(" ".join(cur))
    return lines

def make_subtitle_clip(text: str, duration: float, cfg_render: Dict[str, Any]) -> ImageClip:
    w, h = config["render"].get("vertical_resolution", [V_WIDTH, V_HEIGHT])
    font_path = cfg_render.get("font", r"C:\Windows\Fonts\arial.ttf")
    font_size = int(cfg_render.get("font_size", 52))
    line_h = float(cfg_render.get("line_height", 1.25))
    max_w = int(cfg_render.get("subtitle_max_width_px", int(w*0.86)))
    bottom_margin = int(cfg_render.get("subtitle_bottom_margin_px", 180))
    show_box = bool(cfg_render.get("subtitle_box", True))

    img = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    font = _load_font(font_path, font_size)
    lines = _wrap_text(draw, text, font, max_w)

    line_sizes = [draw.textbbox((0,0), ln, font=font) for ln in lines]
    lh = int(font_size * line_h)
    total_h = lh * max(1, len(lines))
    y0 = h - bottom_margin - total_h

    if show_box:
        padding = 24
        all_w = max((r[2] for r in line_sizes), default=0)
        box_w = min(max_w + padding*2, int(all_w + padding*2))
        box_h = total_h + padding*2
        x_box = (w - box_w)//2
        y_box = y0 - padding
        box = Image.new("RGBA", (box_w, box_h), (0,0,0,140))
        img.alpha_composite(box, (x_box, y_box))

    y = y0
    for ln in lines:
        tw = draw.textbbox((0,0), ln, font=font)[2]
        x = (w - tw)//2
        draw.text((x+2, y+2), ln, font=font, fill=(0,0,0,200))           # shadow
        draw.text((x, y), ln, font=font, fill=(255,255,255,255))         # text
        y += lh

    np_img = np.array(img).astype("uint8")
    clip = ImageClip(np_img).set_duration(max(0.05, float(duration)))
    return clip

# -----------------------------
# Renderer
# -----------------------------
def _load_background_clip(path: Optional[str], duration: float) -> VideoFileClip:
    if path and Path(path).exists():
        try:
            c = VideoFileClip(path)
            if c.duration < duration - 0.05:
                loops = int(math.ceil(duration / max(c.duration, 0.5)))
                parts = []
                while sum((p.duration for p in parts), 0) + 0.01 < duration and len(parts) < loops + 2:
                    sub = c.subclip(0, min(c.duration, duration - sum(p.duration for p in parts)))
                    parts.append(sub)
                c = concatenate_videoclips(parts) if parts else c
            else:
                c = c.subclip(0, duration)
            return c
        except Exception as e:
            console.print(f"[yellow]Background load failed[/yellow]: {e}")
    # fallback color
    return ColorClip(size=(V_WIDTH, V_HEIGHT), color=(0,0,0), duration=duration)

def _fit_vertical(clip: VideoFileClip, target_w: int, target_h: int) -> VideoFileClip:
    w, h = clip.size
    target_ar = target_w / target_h
    ar = w / h
    if ar > target_ar:
        new_h = target_h
        new_w = int(ar * new_h)
        c = vfx_resize(clip, (new_w, new_h))
        x1 = (new_w - target_w)//2
        c = vfx_crop(c, x1=x1, y1=0, x2=x1+target_w, y2=target_h)
    else:
        new_w = target_w
        new_h = int(new_w / ar)
        c = vfx_resize(clip, (new_w, new_h))
        y1 = (new_h - target_h)//2
        c = vfx_crop(c, x1=0, y1=y1, x2=target_w, y2=target_h)
    return c

def _compose_audio(narration: AudioFileClip, music_path: Optional[Path]) -> AudioFileClip:
    if music_path and music_path.exists():
        try:
            music = AudioFileClip(str(music_path))
            music = volumex(music, float(config["render"]["music_volume"]))
            dur = max(narration.duration, music.duration)
            music = music.audio_loop(duration=dur)
            narr = audio_normalize(narration)
            # Mix by simple sum with ducked music
            return CompositeAudioClipSafe([narr.set_duration(dur), volumex(music, float(config["render"]["music_duck"]))])
        except Exception as e:
            console.print(f"[yellow]Music mix failed[/yellow]: {e}")
    return audio_normalize(narration)

# Helper to avoid moviepy lazy-eval pitfalls when summing AudioFileClips
from moviepy.audio.AudioClip import CompositeAudioClip as _CompositeAudioClip
def CompositeAudioClipSafe(clips: List[AudioFileClip]) -> AudioFileClip:
    return _CompositeAudioClip(clips)  # thin wrapper for clarity

def render_video() -> None:
    data = json.load(open(ROOT / "shotlist.json", encoding="utf-8"))
    shots = data["shots"]
    title = data.get("title", "Reddit Story")

    narr_path = Path(config["tts"].get("output", str(AUDIO / "narration.wav")))
    if not narr_path.exists():
        console.print("[red]Missing narration.wav. Run TTS step first.[/red]")
        return
    narration = AudioFileClip(str(narr_path))

    # Load word timings and build mapping
    wjson = AUDIO / "word_timings.json"
    word_timings = []
    if wjson.exists():
        word_timings = json.load(open(wjson, encoding="utf-8")).get("word_timings", [])
    mapping = build_word_mapping(shots, word_timings)

    # Pick a random music track if available
    music_files: List[Path] = []
    for ext in ("*.m4a","*.mp3","*.wav","*.aac","*.ogg"):
        music_files += list((ASSETS / "music").glob(ext))
    music_path = random.choice(music_files) if music_files else None

    # Build per-shot clips
    w_target, h_target = config["render"].get("vertical_resolution", [V_WIDTH, V_HEIGHT])
    target_sz = (w_target, h_target)

    visual_clips = []
    for m in mapping:
        duration = max(0.2, float(m["end"] - m["start"]))
        bg_clip = _load_background_clip(
            next((s["background"] for s in shots if s["idx"] == m["idx"]), None),
            duration
        )
        bg_clip = _fit_vertical(bg_clip, w_target, h_target)

        sub_clip = make_subtitle_clip(m["text"], duration, config["render"])
        sub_clip = sub_clip.set_position(("center", "bottom"))

        comp = CompositeVideoClip([bg_clip, sub_clip], size=target_sz).set_duration(duration)
        visual_clips.append(comp)

    final_v = concatenate_videoclips(visual_clips, method="compose")
    # Align narration to total duration
    total_dur = final_v.duration
    if narration.duration < total_dur:
        narration = narration.audio_loop(duration=total_dur)
    else:
        narration = narration.subclip(0, total_dur)

    final_a = _compose_audio(narration, music_path)
    final = final_v.set_audio(final_a)

    safe_mkdir(OUTPUT)
    ts = int(time.time())
    base = slugify(title, maxlen=50) or "video"
    out_main = OUTPUT / f"{base}_{ts}_1080x1920.mp4"
    out_small = OUTPUT / f"{base}_{ts}_720x1280.mp4"

    final.write_videofile(str(out_main), fps=FPS, threads=4, codec="libx264", audio_codec="aac", preset="medium", bitrate="6M")
    # small version
    final_small = vfx_resize(final, newsize=(V_WIDTH_SMALL, int(V_HEIGHT * (V_WIDTH_SMALL / V_WIDTH))))
    final_small.write_videofile(str(out_small), fps=FPS, threads=4, codec="libx264", audio_codec="aac", preset="faster", bitrate="3M")

    # External subs
    srt_path = OUTPUT / f"{base}_{ts}.srt"
    ass_path = OUTPUT / f"{base}_{ts}.ass"
    write_srt(mapping, srt_path)
    write_ass(mapping, ass_path, res=target_sz)

    console.print(f"[green]Rendered[/green] {out_main.name} and {out_small.name}")
    if config["render"].get("make_shorts", True):
        make_shorts(out_main, base, ts)

def make_shorts(out_path: Path, base: str, ts: int) -> None:
    max_s = int(config["render"].get("shorts_max_seconds", 60))
    try:
        clip = VideoFileClip(str(out_path))
        parts = []
        t = 0.0; i = 1
        while t < clip.duration - 0.3:
            end = min(t + max_s, clip.duration)
            sub = clip.subclip(t, end)
            short_name = OUTPUT / f"{base}_{ts}_short{i:02d}.mp4"
            sub.write_videofile(str(short_name), fps=FPS, threads=4, codec="libx264", audio_codec="aac", preset="faster", bitrate="4M")
            parts.append(short_name)
            i += 1
            t = end
        console.print(f"[green]Shorts created[/green]: {len(parts)}")
    except Exception as e:
        console.print(f"[yellow]Shorts failed[/yellow]: {e}")

# -----------------------------
# Backup and Split
# -----------------------------
def backup_project() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = ROOT / f"backup_{ts}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fname in ["story.txt", "stories_meta.json", "shotlist.json"]:
            p = ROOT / fname
            if p.exists():
                z.write(p, arcname=p.name)
        for folder in [OUTPUT, AUDIO, ASSETS]:
            if folder.exists():
                for f in folder.rglob("*"):
                    if f.is_file():
                        z.write(f, arcname=str(f.relative_to(ROOT)))
        cfg = CONFIG_FILE
        if cfg.exists():
            z.write(cfg, arcname=cfg.name)
    console.print(f"[green]Backup saved[/green] {zip_path.name}")

def split_video(path: Path, max_minutes: int = 10) -> List[Path]:
    clip = VideoFileClip(str(path))
    dur = clip.duration
    parts = []
    i = 0
    while i * max_minutes * 60 < dur - 0.01:
        start = i * max_minutes * 60
        end = min((i+1)*max_minutes*60, dur)
        sub = clip.subclip(start, end)
        out = OUTPUT / f"{path.stem}_part{i+1}.mp4"
        sub.write_videofile(str(out), fps=FPS, codec="libx264", audio_codec="aac")
        parts.append(out)
        i += 1
    return parts

# -----------------------------
# Menus / CLI
# -----------------------------
def download_menu() -> None:
    while True:
        console.print("\n--- Download Menu ---")
        console.print("1. Backgrounds")
        console.print("2. Music")
        console.print("3. SFX")
        console.print("4. ALL")
        console.print("0. Back")
        choice = input("Select option: ").strip()
        if choice == "1":
            download_assets("backgrounds")
        elif choice == "2":
            download_assets("music")
        elif choice == "3":
            download_assets("sfx")
        elif choice == "4":
            for cat in ["backgrounds","music","sfx"]:
                download_assets(cat)
        elif choice == "0":
            break

def main_menu() -> None:
    while True:
        console.print("\n=== RedditVideoBot Main Menu ===")
        console.print("1. Download assets")
        console.print("2. Fetch Reddit stories")
        console.print("3. Generate shotlist")
        console.print("4. Generate TTS")
        console.print("5. Render video")
        console.print("6. Backup project")
        console.print("7. Full pipeline")
        console.print("8. Split a video")
        console.print("0. Exit")
        choice = input("Select option: ").strip()
        if choice == "1":
            download_menu()
        elif choice == "2":
            fetch_reddit_stories()
        elif choice == "3":
            generate_shotlist()
        elif choice == "4":
            tts_generate()
        elif choice == "5":
            render_video()
        elif choice == "6":
            backup_project()
        elif choice == "7":
            pipeline()
        elif choice == "8":
            p = input("Path to video to split: ").strip().strip('\"')
            m = int(input("Max minutes per part [10]: ").strip() or "10")
            split_video(Path(p), max_minutes=m)
        elif choice == "0":
            break

def pipeline() -> None:
    fetch_reddit_stories()
    generate_shotlist()
    tts_generate()
    render_video()
    backup_project()

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    if not ffmpeg_available():
        console.print("[yellow]ffmpeg not detected in PATH. MoviePy may fail.[/yellow]")
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[cyan]Interrupted[/cyan]")
