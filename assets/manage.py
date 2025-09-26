from dataclasses import dataclass
import random
@dataclass
class Asset: path:str; mood:str='any'; weight:float=1.0
def load_assets_from_cfg(cfg): return {'backgrounds':[Asset(**a) for a in cfg.get('assets',{}).get('backgrounds',[])],'music':[Asset(**a) for a in cfg.get('assets',{}).get('music',[])]}
def pick(lst,mood,k=1): return random.choices(lst,k=k)
