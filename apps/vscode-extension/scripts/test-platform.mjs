import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const assetMap = JSON.parse(readFileSync(join(scriptDir, "asset-map.json"), "utf8"));

assert.equal(assetMap["win32-x64"], "waggle-mcp-windows-x64.exe");
assert.equal(assetMap["linux-x64"], "waggle-mcp-linux-x64");
assert.equal(assetMap["darwin-arm64"], "waggle-mcp-darwin-arm64");
assert.equal(assetMap["darwin-x64"], "waggle-mcp-darwin-x64");

console.log("platform asset map ok");
