import * as fs from "fs";
import * as path from "path";

export type PlatformAssetKey = "win32-x64" | "darwin-arm64" | "darwin-x64" | "linux-x64";

const ASSET_MAP_PATH = path.join(__dirname, "..", "scripts", "asset-map.json");

export function platformAssetKey(): PlatformAssetKey {
  const arch = process.arch === "arm64" ? "arm64" : "x64";
  if (process.platform === "win32") {
    return "win32-x64";
  }
  if (process.platform === "darwin") {
    return arch === "arm64" ? "darwin-arm64" : "darwin-x64";
  }
  if (process.platform === "linux") {
    return "linux-x64";
  }
  throw new Error(`Unsupported platform: ${process.platform} ${process.arch}`);
}

export function loadAssetMap(): Record<PlatformAssetKey, string> {
  const raw = fs.readFileSync(ASSET_MAP_PATH, "utf8");
  return JSON.parse(raw) as Record<PlatformAssetKey, string>;
}

export function assetFileNameForCurrentPlatform(): string {
  const key = platformAssetKey();
  const map = loadAssetMap();
  const name = map[key];
  if (!name) {
    throw new Error(`No release binary mapped for ${key}`);
  }
  return name;
}
