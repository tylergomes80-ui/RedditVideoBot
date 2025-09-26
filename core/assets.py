from dataclasses import dataclass
from typing import List, Dict, Optional
import random

@dataclass
class Asset:
    path: str
    mood: str
    weight: float = 1.0
    kind: str = "background"  # or "music"

def pick_assets(assets: List[Asset], target_mood: str, k:int=1) -> List[Asset]:
    pool = [a for a in assets if a.mood==target_mood] or assets
    weights = [max(0.01,a.weight) for a in pool]
    return random.choices(pool, weights=weights, k=k)

def from_config(cfg: Dict) -> Dict[str, List[Asset]]:
    res = {"backgrounds": [], "music": []}
    for item in cfg.get("assets", {}).get("backgrounds", []):
        res["backgrounds"].append(Asset(path=item["path"], mood=item.get("mood","any"), weight=float(item.get("weight",1.0)), kind="background"))
    for item in cfg.get("assets", {}).get("music", []):
        res["music"].append(Asset(path=item["path"], mood=item.get("mood","any"), weight=float(item.get("weight",1.0)), kind="music"))
    return res
