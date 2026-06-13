#!/usr/bin/env python3
"""
Sync or validate version numbers across python, VS Code, and Claude extension manifests.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Paths relative to repository root
FILES = {
    "pyproject.toml": "toml",
    "apps/vscode-extension/package.json": "json",
    "apps/vscode-extension/package-lock.json": "json_lock",
    "apps/mcp/claude-desktop-extension/package.json": "json",
    "apps/mcp/claude-desktop-extension/package-lock.json": "json_lock",
    "apps/mcp/claude-desktop-extension/manifest.json": "json",
}

def read_toml_version(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError(f"Could not find version in {path}")
    return match.group(1)

def write_toml_version(path: Path, version: str) -> None:
    content = path.read_text(encoding="utf-8")
    new_content, count = re.subn(r'(?m)^version\s*=\s*"[^"]+"', f'version = "{version}"', content)
    if count == 0:
        raise ValueError(f"Could not find version line to replace in {path}")
    path.write_text(new_content, encoding="utf-8")

def read_json_version(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["version"]

def write_json_version(path: Path, version: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["version"] = version
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

def read_json_lock_version(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["version"]

def write_json_lock_version(path: Path, version: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["version"] = version
    if "packages" in data and "" in data["packages"]:
        data["packages"][""]["version"] = version
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

def read_version(rel_path: str, file_type: str) -> str:
    path = REPO_ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if file_type == "toml":
        return read_toml_version(path)
    elif file_type == "json":
        return read_json_version(path)
    elif file_type == "json_lock":
        return read_json_lock_version(path)
    else:
        raise ValueError(f"Unknown file type: {file_type}")

def write_version(rel_path: str, file_type: str, version: str) -> None:
    path = REPO_ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if file_type == "toml":
        write_toml_version(path, version)
    elif file_type == "json":
        write_json_version(path, version)
    elif file_type == "json_lock":
        write_json_lock_version(path, version)
    else:
        raise ValueError(f"Unknown file type: {file_type}")

def check_versions() -> bool:
    versions = {}
    has_error = False
    
    for rel_path, file_type in FILES.items():
        try:
            version = read_version(rel_path, file_type)
            versions[rel_path] = version
        except Exception as e:
            print(f"Error reading {rel_path}: {e}")
            has_error = True
            
    if has_error:
        return False
        
    unique_versions = set(versions.values())
    if len(unique_versions) > 1:
        print("Mismatched versions found:")
        for rel_path, version in versions.items():
            print(f"  {rel_path}: {version}")
        return False
        
    print(f"All files are in sync at version: {list(unique_versions)[0]}")
    return True

def sync_versions(target_version: str = None) -> bool:
    if target_version is None:
        try:
            target_version = read_version("pyproject.toml", "toml")
            print(f"Read single source of truth version '{target_version}' from pyproject.toml")
        except Exception as e:
            print(f"Error reading pyproject.toml: {e}")
            return False
            
    print(f"Syncing all files to version: {target_version}")
    for rel_path, file_type in FILES.items():
        try:
            write_version(rel_path, file_type, target_version)
            print(f"  Updated {rel_path}")
        except Exception as e:
            print(f"Error writing to {rel_path}: {e}")
            return False
            
    return True

def main():
    parser = argparse.ArgumentParser(description="Sync or check versions across configuration manifests.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Check that all manifest versions are identical.")
    group.add_argument("--sync", nargs="?", const="", help="Sync versions. If a version is provided, syncs to that. If empty, syncs other files to pyproject.toml version.")
    
    args = parser.parse_args()
    
    if args.check:
        if not check_versions():
            sys.exit(1)
    elif args.sync is not None:
        target = args.sync if args.sync != "" else None
        if not sync_versions(target):
            sys.exit(1)

if __name__ == "__main__":
    main()
