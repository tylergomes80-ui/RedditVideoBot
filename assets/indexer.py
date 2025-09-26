import os
import json

def scan_sources(paths):
    assets = []
    for path in paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.lower().endswith(".mp4"):
                    assets.append({
                        "path": os.path.join(path, file),
                        "tags": ["video"]
                    })
    return assets

if __name__ == "__main__":
    index = scan_sources(["./assets/backgrounds"])
    with open("assets_index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"Indexed {len(index)} assets.")
