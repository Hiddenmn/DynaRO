"""Standard-library verification; no array, model or evaluation is loaded."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re


def require(value, message):
    if not value: raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def safe_file(root,name):
    require(isinstance(name,str) and name and ":" not in name and "\\" not in name
            and all(p not in ("",".","..") for p in name.split("/")),"Unsafe relative input")
    root=Path(root).resolve(strict=True)
    path=root/name
    require(path.is_file() and path.resolve().is_relative_to(root)
            and not any(p.is_symlink() or p.is_junction() for p in (path,*path.parents) if p!=root),
            "Missing or redirected input: "+name)
    return path


def private_text(text):
    patterns=(r"(?i)(?<![a-z])[a-z]:[\\/]",r"(?i)/(?:Users|home|root|autodl-tmp)/",
              r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+)")
    require(not any(re.search(p,text) for p in patterns),"Private path/contact/credential-like text")


def verify(root):
    root=Path(root).resolve(strict=True)
    m=json.loads(safe_file(root,"manifest.json").read_text(encoding="utf-8"))
    require(m["format"]=="dynaro-evaluation-candidate-v1" and m["public_release"] is False
            and m["licence"] is None,"Different evaluation component")
    private_text(json.dumps(m))
    require({p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
            ==set(m["files"])|{"manifest.json"},"Unexpected/missing evaluation file")
    for name,row in m["files"].items():
        path=safe_file(root,name)
        require(path.stat().st_size==row["bytes"] and digest(path)==row["sha256"],"Changed file: "+name)
        raw=path.read_text(encoding="utf-8-sig")
        private_text(raw)
        if name.endswith(".py"):compile(raw,name,"exec")
    return {"status":"evaluation_sources_verified","files":len(m["files"]),
            "model_calls":0,"pairs_scored":0,"public_release":False}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    print(json.dumps(verify(p.parse_args().root),indent=2))
