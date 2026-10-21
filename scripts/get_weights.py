"""Download the Depth Anything V2 weights used to initialise training."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

REPO = {
    "vits": "Depth-Anything-V2-Small",
    "vitb": "Depth-Anything-V2-Base",
    "vitl": "Depth-Anything-V2-Large",
}


def main():
    ap = argparse.ArgumentParser(description="Fetch base Depth Anything V2 weights.")
    ap.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--out_dir", default="checkpoints/DepthAnythingV2")
    args = ap.parse_args()

    fn = f"depth_anything_v2_{args.encoder}.pth"
    dst = os.path.join(args.out_dir, fn)
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print(f"[weights] already present: {dst} ({os.path.getsize(dst)/2**30:.2f} GB) - skipping")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    try:
        import truststore
        truststore.inject_into_ssl()
        print("[weights] truststore injected (OS trust store)")
    except Exception:
        pass

    url = f"https://huggingface.co/depth-anything/{REPO[args.encoder]}/resolve/main/{fn}"
    tmp = dst + ".part"
    print(f"[weights] downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "pitchdepth"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r[weights] {done/2**20:6.0f} / {total/2**20:.0f} MB ({pct:4.1f}%)", end="")
            print()
        os.replace(tmp, dst)
        print(f"[weights] saved {dst} ({os.path.getsize(dst)/2**30:.2f} GB)")
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"[weights] DOWNLOAD FAILED: {exc}\n"
              f"          Download manually to {dst} from:\n          {url}")
        sys.exit(1)


if __name__ == "__main__":
    main()
