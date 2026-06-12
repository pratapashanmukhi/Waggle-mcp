# Codex

Use this when you want Waggle connected to Codex as a local stdio MCP server.

Waggle is local graph memory for coding agents.

No cloud account. No API key. Local by default.

## One-line install

```bash
pipx install waggle-mcp
waggle-mcp setup --yes
```

`waggle-mcp setup --yes` writes a managed Waggle memory block into `AGENTS.md` in the current workspace so Codex can use Waggle from that repo.

### Managed `AGENTS.md` Block

When run inside a workspace, the setup command inserts a managed section inside the `AGENTS.md` file wrapped in specific HTML comment delimiters:

```markdown
<!-- waggle:auto-memory:start -->
## Waggle Automatic Memory
...
<!-- waggle:auto-memory:end -->
```

* **What it is for**: This block provides instructions telling AI agents (like Codex or Antigravity) to automatically call Waggle tools (`prime_context`, `query_graph`, `observe_conversation`) during active chat threads rather than requiring manual user actions.
* **Do not edit manually**: Do not manually modify any text inside the `<!-- waggle:auto-memory:start -->` and `<!-- waggle:auto-memory:end -->` delimiters. Any manual changes inside this block will be overwritten when `waggle-mcp setup --yes` or `waggle-mcp init` is run again.
* **What is safe to customize**: You can add your own custom rules, project descriptions, or team conventions anywhere *outside* this block (either above the start marker or below the end marker). These custom instructions are completely safe and will not be touched by Waggle.

For more details on how these rules govern agent behavior, see the [Automatic Memory Rules Guide](../automatic-memory-rules.md).

## Manual config

Add Waggle to `~/.codex/config.toml`:

```toml
[mcp_servers.waggle]
command = "waggle-mcp"
args = ["serve", "--transport", "stdio"]
env = {
  WAGGLE_BACKEND = "sqlite",
  WAGGLE_DB_PATH = "~/.waggle/waggle.db",
  WAGGLE_DEFAULT_TENANT_ID = "local-default",
  WAGGLE_MODEL = "all-MiniLM-L6-v2"
}
```

A pre-filled example is available at
[`examples/codex_config.example.toml`](../../examples/codex_config.example.toml).

## Verify

```bash
waggle-mcp doctor
```

Restart Codex and confirm Waggle tools such as `prime_context`, `query_graph`,
and `observe_conversation` are available.

## Troubleshooting

See [troubleshooting.md](./troubleshooting.md).

## Security and privacy

Waggle stores memory locally by default in SQLite. Set `WAGGLE_DB_PATH`
explicitly if you want Codex and other MCP clients to share the same local
memory graph.
