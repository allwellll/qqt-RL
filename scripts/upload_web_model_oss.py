"""Upload selected immutable model assets under web-models/, then verify public bytes.

Credentials are environment-only. Bucket ACL and CORS are checked, never changed.
Use --verify-only to validate an existing model without uploading anything.
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import urllib.request


def transfer(catalog, model_id, asset_dir, verify_only=False):
    import oss2
    logging.getLogger("oss2").setLevel(logging.CRITICAL)
    document = json.loads(Path(catalog).read_text())
    rows = [row for row in document["models"] if row["id"] == model_id]
    if len(rows) != 1:
        raise ValueError("select exactly one model")
    row = rows[0]
    assets = [row] + ([row["onnx"]] if row.get("onnx") else [])
    for asset in assets:
        name, sha = asset["file"], asset["sha256"]
        if (not re.fullmatch(r"[a-z0-9][a-z0-9._-]*\.(json|onnx)", name)
                or not re.fullmatch(r"[0-9a-f]{64}", sha) or sha not in name):
            raise ValueError("immutable neutral SHA256 filename required")
        local = Path(asset_dir) / name
        if local.stat().st_size != asset["bytes"] or hashlib.sha256(local.read_bytes()).hexdigest() != sha:
            raise ValueError("local model SHA256/size mismatch")
    auth = oss2.Auth(os.environ["QQT_OSS_RW_AK_ID"], os.environ["QQT_OSS_RW_AK_SECRET"])
    endpoint, bucket_name = os.environ["QQT_OSS_ENDPOINT"], os.environ["QQT_OSS_BUCKET"]
    if bucket_name != "qqt-model" or endpoint.rstrip("/").split("://")[-1] != "oss-cn-beijing.aliyuncs.com":
        raise ValueError("unexpected model bucket/endpoint")
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    if bucket.get_bucket_acl().acl != "private":
        raise ValueError("model bucket must remain private")
    allowed = {"https://allwellll.github.io", "http://localhost:*", "http://127.0.0.1:*"}
    cors = bucket.get_bucket_cors().rules
    if (not cors or {origin for rule in cors for origin in rule.allowed_origins} != allowed
            or any(set(rule.allowed_methods) - {"GET", "HEAD"} for rule in cors)):
        raise ValueError("unexpected model CORS scope")
    verified = []
    for asset in assets:
        key = "web-models/" + asset["file"]
        local = Path(asset_dir) / asset["file"]
        mime = "application/json" if key.endswith(".json") else "application/octet-stream"
        cache = "public, max-age=31536000, immutable"
        if not verify_only:
            try:
                head = bucket.head_object(key)
                if head.content_length != asset["bytes"]:
                    raise ValueError("immutable object collision")
            except oss2.exceptions.NoSuchKey:
                bucket.put_object_from_file(key, str(local), headers={
                    "Content-Type": mime, "Cache-Control": cache,
                    "x-oss-meta-sha256": asset["sha256"], "x-oss-object-acl": "public-read",
                    "x-oss-forbid-overwrite": "true"})
            if bucket.get_object_acl(key).acl != "public-read":
                bucket.put_object_acl(key, oss2.OBJECT_ACL_PUBLIC_READ)
        head = bucket.head_object(key)
        if (head.content_length != asset["bytes"] or bucket.get_object_acl(key).acl != "public-read"
                or head.headers.get("Content-Type") != mime or head.headers.get("Cache-Control") != cache):
            raise ValueError("public object metadata mismatch")
        url = "https://qqt-model.oss-cn-beijing.aliyuncs.com/" + key
        hasher, size = hashlib.sha256(), 0
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "QQT-Model-Verification"}), timeout=90) as response:
            if response.status != 200:
                raise ValueError("public object unavailable")
            while chunk := response.read(1024 * 1024):
                hasher.update(chunk)
                size += len(chunk)
        if size != asset["bytes"] or hasher.hexdigest() != asset["sha256"]:
            raise ValueError("public model SHA256/size mismatch")
        verified.append(dict(key=key, bytes=size, sha256=hasher.hexdigest(), acl="public-read"))
    if bucket.get_bucket_acl().acl != "private":
        raise ValueError("bucket ACL changed")
    return dict(status="PASS", model_id=model_id, verify_only=verify_only,
                bucket_acl="private", cors_origins=sorted(allowed), objects=verified)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--asset-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    try:
        result = transfer(a.catalog, a.model_id, a.asset_dir, a.verify_only)
        Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    except Exception as error:
        # SDK exception text can include request credentials; never serialize it.
        print(json.dumps(dict(status="FAIL", error_type=type(error).__name__)))
        raise SystemExit(1)
