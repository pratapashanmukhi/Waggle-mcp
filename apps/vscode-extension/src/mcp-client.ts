export async function callStdioTool(
  command: string,
  env: Record<string, string>,
  toolName: string,
  args: Record<string, unknown>
): Promise<unknown> {
  const { Client } = await import("@modelcontextprotocol/sdk/client/index.js");
  const { StdioClientTransport } = await import("@modelcontextprotocol/sdk/client/stdio.js");
  const transport = new StdioClientTransport({
    command,
    args: ["serve", "--transport", "stdio"],
    env: { ...process.env, ...env } as Record<string, string>,
    stderr: "pipe"
  });
  const client = new Client({ name: "waggle-vscode", version: "1.0.0" }, { capabilities: {} });
  await client.connect(transport);
  try {
    const result = await client.callTool({ name: toolName, arguments: args });
    return result;
  } finally {
    await transport.close();
  }
}
