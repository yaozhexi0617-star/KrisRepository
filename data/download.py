# -*- coding: utf-8 -*-
"""下载并解压 MovieLens latest-small。

用法：
    python -m data.download          # 已有则跳过
    python -m data.download --force  # 重新下载
"""
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CACHE_DIR, ML_SUBDIR, ML_URL, RAW_DIR  # noqa: E402


def _context():
    """优先用 certifi 的证书链；某些代理环境需要它。"""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def download(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if ML_SUBDIR.exists() and not force:
        print(f"[skip] 已存在：{ML_SUBDIR}")
        return ML_SUBDIR

    zip_path = CACHE_DIR / "ml-latest-small.zip"
    if force or not zip_path.exists():
        print(f"[get ] {ML_URL}")
        req = urllib.request.Request(ML_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180, context=_context()) as r, \
                open(zip_path, "wb") as f:
            f.write(r.read())
        print(f"[ok  ] {zip_path.stat().st_size / 1e3:.0f} KB -> {zip_path}")

    print(f"[unzip] -> {RAW_DIR}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)
    print(f"[ok  ] {ML_SUBDIR}")
    return ML_SUBDIR


if __name__ == "__main__":
    download(force="--force" in sys.argv)
