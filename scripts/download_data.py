"""Download the MoRTH 'Road Accidents in India 2024' report (published via OpenCity)."""
from pathlib import Path

import requests

URL = (
    "https://data.opencity.in/dataset/33d29ab0-f9e8-4fc7-b404-c93c1ed8e1b8/resource/"
    "30af828c-3513-4c74-a919-8daa708f077d/download/road-accidents-in-india-2024.pdf"
)
DEST = Path("data/raw/road-accidents-in-india-2024.pdf")


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    DEST.write_bytes(resp.content)
    print(f"Saved {len(resp.content) / 1e6:.1f} MB -> {DEST}")


if __name__ == "__main__":
    main()
