import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

export async function callStdioTool(
  command: string,
  env: Record<string, string>,
  toolName: string,
  args: Record<string, unknown>
): Promise<unknown> {
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
