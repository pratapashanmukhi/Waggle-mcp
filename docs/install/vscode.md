# VS Code

Use the Waggle VS Code extension for one-click workspace setup with a bundled release binary (no pip install required by default).

## One-line install

Install the Marketplace extension, open a workspace, then run:

**Waggle: Enable for this Workspace**

## Default behavior

- Downloads `waggle-mcp` from [GitHub Releases](https://github.com/Abhigyan-Shekhar/Waggle-mcp/releases) for your platform
- Starts a local HTTP server (`graph-studio`) when the workspace opens
- Writes `.vscode/mcp.json` for agent MCP over stdio (after you confirm)

## Manual MCP config

If you prefer not to use the extension, add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "waggle": {
      "type": "stdio",
      "command": "waggle-mcp",
      "args": ["serve", "--transport", "stdio"],
      "env": {
        "WAGGLE_DEFAULT_TENANT_ID": "${workspaceFolderBasename}",
        "WAGGLE_DB_PATH": "~/.waggle/waggle.db",
        "WAGGLE_STARTUP_MODE": "fast"
      }
    }
  }
}
```

With the extension’s binary install, `command` is the cached executable path under VS Code global storage.

## pipx fallback

Set `waggle.installMethod` to `pipx` in VS Code settings if you already use `pipx install waggle-mcp`.

## Verify

```bash
waggle-mcp doctor
```

Or use **Waggle: Run Doctor** in the command palette.

Reload VS Code, switch the agent UI into MCP-capable mode, and confirm the Waggle server is enabled.

## Troubleshooting

- **Binary download fails** — check `waggle.binaryReleaseRepo` and network; try `waggle.binaryVersion` matching an existing tag (e.g. `0.1.15`).
- **Antivirus blocks the binary** — allow the file under VS Code global storage or use `pipx`.
- See [troubleshooting.md](./troubleshooting.md).

## Security and privacy

Workspace MCP config is visible in the repo. That keeps the command path, tenant ID, and database path auditable during code review.
