import * as vscode from "vscode";
import { BinaryResolver } from "./binary-resolver";
import { registerWaggleCommands, type WaggleStatus } from "./commands";
import { ServerManager } from "./server-manager";
import { resolveTenantId } from "./mcp-config";
import { GraphStudioViewProvider } from "./graph-studio-view";

const OUTPUT_CHANNEL = "Waggle";
const DEFAULT_DB_PATH = "~/.waggle/waggle.db";

let activeServerManager: ServerManager | undefined;

export function activate(context: vscode.ExtensionContext): void {
  const output = vscode.window.createOutputChannel(OUTPUT_CHANNEL);
  const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBar.command = "waggle.showStatus";

  const append = (message: string): void => {
    output.appendLine(`[waggle] ${message}`);
  };

  const config = (): vscode.WorkspaceConfiguration => vscode.workspace.getConfiguration("waggle");
  const workspaceFolder = (): vscode.WorkspaceFolder | undefined => vscode.workspace.workspaceFolders?.[0];

  const setStatus = (status: WaggleStatus, detail = ""): void => {
    const suffix = detail ? `: ${detail}` : "";
    const labels: Record<WaggleStatus, string> = {
      "not-installed": `Waggle: Not Installed${suffix}`,
      ready: `Waggle: Ready${suffix}`,
      connected: `Waggle: Connected${suffix}`,
      error: `Waggle: Error${suffix}`
    };
    statusBar.text = labels[status];
    statusBar.show();
  };

  const serverEnv = (): Record<string, string> => {
    const folder = workspaceFolder();
    return {
      WAGGLE_DEFAULT_TENANT_ID: resolveTenantId(folder, config()),
      WAGGLE_DB_PATH: config().get<string>("dbPath", DEFAULT_DB_PATH),
      WAGGLE_STARTUP_MODE: "fast",
      WAGGLE_MODEL: config().get<string>("model", "deterministic")
    };
  };

  const resolver = new BinaryResolver(context);
  const serverManager = new ServerManager(context, resolver, append);
  activeServerManager = serverManager;
  const graphView = new GraphStudioViewProvider(serverManager);

  context.subscriptions.push(
    output,
    statusBar,
    serverManager.onDidChange(() => graphView.refresh()),
    vscode.window.registerWebviewViewProvider(GraphStudioViewProvider.viewType, graphView),
    { dispose: () => void serverManager.stop() }
  );

  registerWaggleCommands(
    {
      output,
      statusBar,
      resolver,
      serverManager,
      append,
      setStatus,
      config,
      workspaceFolder,
      serverEnv
    },
    context.subscriptions
  );
}

export function deactivate(): void {
  void activeServerManager?.stop();
  activeServerManager = undefined;
}
