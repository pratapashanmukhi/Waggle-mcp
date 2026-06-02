# Waggle: Local Memory for AI Agents

Local graph memory for coding agents using MCP.

Waggle gives VS Code agents persistent repo memory without a cloud account or API key.

## What it does

- Downloads a standalone `waggle-mcp` binary from GitHub Releases (no pip/Python required by default)
- Auto-starts a local HTTP server when a workspace opens (`waggle.autoStart`)
- Writes `.vscode/mcp.json` for agent MCP (stdio transport) after you confirm
- **Waggle: Query Memory** — search the graph via the local HTTP API
- **Waggle: Observe Conversation** — persist a user/assistant turn via MCP
- Graph Studio in the activity bar sidebar and via **Open Graph Studio**
- `pipx` install remains available (`waggle.installMethod`: `pipx`)

## Requirements

- VS Code `1.96+`
- Network access on first run (to download the release binary for your OS)
- Optional: `pipx` + Python `3.11+` if you use `installMethod: pipx`

## Quick start

1. Install the extension
2. Open a workspace folder
3. Run **Waggle: Enable for this Workspace** (or accept the startup prompt)
4. Reload VS Code if your MCP consumer requires it

## Commands

- **Waggle: Enable for this Workspace**
- **Waggle: Install Waggle** (download binary or pipx)
- **Waggle: Run Doctor**
- **Waggle: Query Memory**
- **Waggle: Observe Conversation**
- **Waggle: Open Graph Studio**
- **Waggle: Show Status**
- **Waggle: Export Memory**
- **Waggle: Open Install Docs**

## Settings

| Setting | Default | Purpose |
|---------|---------|---------|
| `waggle.installMethod` | `binary` | `binary` or `pipx` |
| `waggle.autoStart` | `true` | Start HTTP server on workspace open |
| `waggle.binaryReleaseRepo` | `Abhigyan-Shekhar/Waggle-mcp` | GitHub repo for release assets |
| `waggle.binaryVersion` | *(extension version)* | Pin a specific release tag |
| `waggle.dbPath` | `~/.waggle/waggle.db` | SQLite path |
| `waggle.tenantId` | workspace folder name | Tenant / project scope |

## Development

```bash
cd apps/vscode-extension
npm install
npm run compile
npm test
```

Press F5 to launch the Extension Development Host.

### Bundled binary layout

Release CI publishes `waggle-mcp-{platform}` assets and `bundle-metadata.json`. The extension maps your OS/arch via `scripts/asset-map.json` and caches under global storage.

## Packaging

```bash
npm run package
```

Binaries are **not** embedded in the VSIX; they are downloaded on first use from GitHub Releases.

## Privacy

- Local by default
- No cloud account required
- Memory stored in `WAGGLE_DB_PATH`
