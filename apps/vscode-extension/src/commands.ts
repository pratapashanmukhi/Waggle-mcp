import * as vscode from "vscode";
import { callStdioTool } from "./mcp-client";
import { execFileAsync } from "./exec";
import { BinaryResolver } from "./binary-resolver";
import { ServerManager } from "./server-manager";
import { resolveTenantId, writeWorkspaceMcpConfig } from "./mcp-config";

export type WaggleStatus = "not-installed" | "ready" | "connected" | "error";

export interface WaggleContext {
  output: vscode.OutputChannel;
  statusBar: vscode.StatusBarItem;
  resolver: BinaryResolver;
  serverManager: ServerManager;
  append: (message: string) => void;
  setStatus: (status: WaggleStatus, detail?: string) => void;
  config: () => vscode.WorkspaceConfiguration;
  workspaceFolder: () => vscode.WorkspaceFolder | undefined;
  serverEnv: () => Record<string, string>;
}

export function registerWaggleCommands(ctx: WaggleContext, disposables: vscode.Disposable[]): void {
  const commandPath = async (): Promise<string> => ctx.resolver.resolveCommandPath();

  const showOutput = (): void => ctx.output.show(true);

  const flushExec = (result: { stdout: string; stderr: string }): void => {
    if (result.stdout.trim()) {
      ctx.output.append(result.stdout);
    }
    if (result.stderr.trim()) {
      ctx.output.append(result.stderr);
    }
  };

  const updateStatusFromEnvironment = async (): Promise<boolean> => {
    try {
      const cmd = await commandPath();
      const result = await execFileAsync(cmd, ["--version"]);
      if (result.code === 0) {
        ctx.setStatus(ctx.serverManager.runtime ? "connected" : "ready", result.stdout.trim());
        return true;
      }
      ctx.setStatus("error", "version check failed");
      flushExec(result);
      return false;
    } catch (error) {
      ctx.setStatus("not-installed");
      ctx.append(`CLI not available: ${String(error)}`);
      return false;
    }
  };

  const ensureServer = async (): Promise<string> => {
    const runtime = ctx.serverManager.runtime;
    if (runtime) {
      return runtime.baseUrl;
    }
    const folder = ctx.workspaceFolder();
    const started = await ctx.serverManager.start(ctx.serverEnv(), folder?.uri.fsPath);
    await ctx.serverManager.writePortFile(started.port);
    ctx.setStatus("connected", `port ${started.port}`);
    return started.baseUrl;
  };

  const runDoctorInternal = async (showSuccessMessage = true): Promise<boolean> => {
    showOutput();
    const cmd = await commandPath();
    ctx.append(`Running: ${cmd} doctor`);
    try {
      const result = await execFileAsync(cmd, ["doctor"], {
        cwd: ctx.workspaceFolder()?.uri.fsPath,
        env: { ...process.env, ...ctx.serverEnv() }
      });
      flushExec(result);
      if (result.code === 0) {
        ctx.setStatus("connected", "doctor ok");
        if (showSuccessMessage) {
          void vscode.window.showInformationMessage("Waggle doctor completed successfully.");
        }
        return true;
      }
      ctx.setStatus("error", "doctor warnings");
      void vscode.window.showWarningMessage("Waggle doctor reported issues. See the Waggle output channel for details.");
      return false;
    } catch (error) {
      ctx.setStatus("error", "doctor failed");
      ctx.append(`Doctor failed: ${String(error)}`);
      void vscode.window.showErrorMessage("Could not run waggle-mcp doctor.");
      return false;
    }
  };

  const installWaggle = async (showPostInstallMessage = true): Promise<boolean> => {
    const method = ctx.config().get<string>("installMethod", "binary");
    if (method === "binary") {
      showOutput();
      try {
        await ctx.resolver.ensureBinary();
        await updateStatusFromEnvironment();
        if (showPostInstallMessage) {
          void vscode.window.showInformationMessage("Waggle binary is ready.");
        }
        return true;
      } catch (error) {
        ctx.setStatus("error", "binary install failed");
        ctx.append(String(error));
        void vscode.window.showErrorMessage("Could not download the Waggle binary. See the output channel.");
        return false;
      }
    }

    showOutput();
    ctx.append("Running: pipx install waggle-mcp");
    try {
      const result = await execFileAsync("pipx", ["install", "waggle-mcp"], {
        cwd: ctx.workspaceFolder()?.uri.fsPath
      });
      flushExec(result);
      if (result.code !== 0) {
        ctx.setStatus("error", "install failed");
        void vscode.window.showErrorMessage("Waggle install failed. See the Waggle output channel for details.");
        return false;
      }
      ctx.append("Waggle installed successfully.");
      await updateStatusFromEnvironment();
      if (showPostInstallMessage) {
        void vscode.window.showInformationMessage("Waggle installed successfully.");
      }
      return true;
    } catch (error) {
      ctx.setStatus("error", "install failed");
      ctx.append(`Install failed: ${String(error)}`);
      void vscode.window.showErrorMessage("Waggle install failed. Ensure pipx is installed and available on PATH.");
      return false;
    }
  };

  const writeWorkspaceConfig = async (): Promise<boolean> => {
    const folder = ctx.workspaceFolder();
    if (!folder) {
      void vscode.window.showWarningMessage("Open a workspace folder before enabling Waggle for this workspace.");
      return false;
    }
    const cmd = await commandPath();
    return writeWorkspaceMcpConfig({
      folder,
      command: cmd,
      tenantId: resolveTenantId(folder, ctx.config()),
      dbPath: ctx.config().get<string>("dbPath", "~/.waggle/waggle.db"),
      mcpConfigScope: ctx.config().get<"servers" | "mcpServers">("mcpConfigScope", "servers"),
      appendLog: ctx.append
    });
  };

  const onboardWaggle = async (): Promise<void> => {
    const folder = ctx.workspaceFolder();
    if (!folder) {
      void vscode.window.showWarningMessage("Open a workspace folder before running Waggle setup.");
      return;
    }

    const proceed = await vscode.window.showInformationMessage(
      "Enable Waggle for this workspace? This installs or downloads Waggle, writes .vscode/mcp.json after confirmation, and runs doctor.",
      { modal: true },
      "Enable Waggle"
    );
    if (proceed !== "Enable Waggle") {
      return;
    }

    const method = ctx.config().get<string>("installMethod", "binary");
    if (method === "binary") {
      const installed = await installWaggle(false);
      if (!installed) {
        return;
      }
    } else {
      const available = await updateStatusFromEnvironment();
      if (!available) {
        const installed = await installWaggle(false);
        if (!installed) {
          return;
        }
      }
    }

    const configured = await writeWorkspaceConfig();
    if (!configured) {
      return;
    }

    if (ctx.config().get<boolean>("autoStart", true)) {
      try {
        await ensureServer();
      } catch (error) {
        ctx.append(`Auto-start failed: ${String(error)}`);
      }
    }

    const doctorOk = await runDoctorInternal(false);
    if (doctorOk) {
      ctx.setStatus("connected", folder.name);
      void vscode.window.showInformationMessage("Waggle is installed, configured, and ready for this workspace.");
      return;
    }
    void vscode.window.showWarningMessage("Waggle was installed and configured, but doctor reported issues. See the Waggle output channel.");
  };

  const queryMemory = async (): Promise<void> => {
    const query = await vscode.window.showInputBox({
      title: "Waggle: Query Memory",
      prompt: "Natural language or ABHI query text",
      placeHolder: "What do we know about authentication?"
    });
    if (!query?.trim()) {
      return;
    }

    showOutput();
    try {
      const baseUrl = await ensureServer();
      const folder = ctx.workspaceFolder();
      const project = resolveTenantId(folder, ctx.config());
      const response = await fetch(`${baseUrl}/api/graph/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project,
          query: query.trim()
        })
      });
      const payload = (await response.json()) as { message?: string; error?: string; [key: string]: unknown };
      if (!response.ok) {
        throw new Error(payload.message || payload.error || `${response.status} ${response.statusText}`);
      }
      ctx.append(JSON.stringify(payload, null, 2));
      void vscode.window.showInformationMessage("Waggle query completed. See the output channel for results.");
    } catch (error) {
      ctx.append(`Query failed: ${String(error)}`);
      void vscode.window.showErrorMessage("Waggle query failed. See the output channel.");
    }
  };

  const observeConversation = async (): Promise<void> => {
    const userMessage = await vscode.window.showInputBox({
      title: "Waggle: Observe Conversation",
      prompt: "User message"
    });
    if (!userMessage?.trim()) {
      return;
    }
    const assistantResponse = await vscode.window.showInputBox({
      title: "Waggle: Observe Conversation",
      prompt: "Assistant response"
    });
    if (!assistantResponse?.trim()) {
      return;
    }

    showOutput();
    try {
      const cmd = await commandPath();
      const folder = ctx.workspaceFolder();
      const project = resolveTenantId(folder, ctx.config());
      const env = ctx.serverEnv();
      const result = await callStdioTool(cmd, env, "observe_conversation", {
        user_message: userMessage.trim(),
        assistant_response: assistantResponse.trim(),
        project
      });
      ctx.append(JSON.stringify(result, null, 2));
      void vscode.window.showInformationMessage("Waggle recorded the conversation turn.");
    } catch (error) {
      ctx.append(`Observe failed: ${String(error)}`);
      void vscode.window.showErrorMessage("Could not observe conversation. See the output channel.");
    }
  };

  const openGraphStudio = async (): Promise<void> => {
    try {
      const baseUrl = await ensureServer();
      const url = `${baseUrl}/graph?mode=edit`;
      ctx.setStatus("connected", "Graph Studio");
      await vscode.env.openExternal(vscode.Uri.parse(url));
    } catch (error) {
      ctx.setStatus("error", "graph studio failed");
      ctx.append(String(error));
      void vscode.window.showErrorMessage("Could not start Waggle Graph Studio.");
    }
  };

  const exportMemory = async (): Promise<void> => {
    const folder = ctx.workspaceFolder();
    const defaultUri = folder ? vscode.Uri.file(`${folder.uri.fsPath}/waggle-export.abhi`) : undefined;
    const target = await vscode.window.showSaveDialog({
      defaultUri,
      filters: { "ABHI Export": ["abhi"] },
      saveLabel: "Export Waggle Memory"
    });
    if (!target) {
      return;
    }

    showOutput();
    const cmd = await commandPath();
    ctx.append(`Running: ${cmd} export --output ${target.fsPath}`);
    try {
      const result = await execFileAsync(cmd, ["export", "--output", target.fsPath], {
        cwd: folder?.uri.fsPath,
        env: { ...process.env, ...ctx.serverEnv() }
      });
      flushExec(result);
      if (result.code !== 0) {
        ctx.setStatus("error", "export failed");
        void vscode.window.showErrorMessage("Waggle export failed. See the output channel for details.");
        return;
      }
      void vscode.window.showInformationMessage(`Waggle memory exported to ${target.fsPath}.`);
    } catch (error) {
      ctx.setStatus("error", "export failed");
      ctx.append(`Export failed: ${String(error)}`);
      void vscode.window.showErrorMessage("Could not export Waggle memory.");
    }
  };

  const openInstallDocs = async (): Promise<void> => {
    await vscode.env.openExternal(
      vscode.Uri.parse("https://github.com/Abhigyan-Shekhar/Waggle-mcp/tree/main/docs/install")
    );
  };

  const showStatus = async (): Promise<void> => {
    await updateStatusFromEnvironment();
    showOutput();
    const runtime = ctx.serverManager.runtime;
    ctx.append(`Status: ${ctx.statusBar.text}`);
    if (runtime) {
      ctx.append(`HTTP server: ${runtime.baseUrl}`);
    }
  };

  const maybePromptInstall = async (): Promise<void> => {
    const method = ctx.config().get<string>("installMethod", "binary");
    let available = false;

    if (method === "binary") {
      if (await ctx.resolver.hasCachedBinary()) {
        available = await updateStatusFromEnvironment();
      }
      if (ctx.config().get<boolean>("autoStart", true) && ctx.workspaceFolder()) {
        try {
          await ensureServer();
          available = true;
        } catch (error) {
          ctx.append(`Background start failed: ${String(error)}`);
        }
      }
    } else {
      available = await updateStatusFromEnvironment();
    }

    if (available) {
      return;
    }

    const choice = await vscode.window.showInformationMessage(
      "Waggle is not set up in this VS Code workspace. Enable it now?",
      "Enable Waggle",
      "Open Docs"
    );
    if (choice === "Enable Waggle") {
      await onboardWaggle();
    } else if (choice === "Open Docs") {
      await openInstallDocs();
    }
  };

  disposables.push(
    vscode.commands.registerCommand("waggle.enableWorkspace", onboardWaggle),
    vscode.commands.registerCommand("waggle.install", () => installWaggle(true)),
    vscode.commands.registerCommand("waggle.doctor", () => runDoctorInternal(true)),
    vscode.commands.registerCommand("waggle.openGraphStudio", openGraphStudio),
    vscode.commands.registerCommand("waggle.showStatus", showStatus),
    vscode.commands.registerCommand("waggle.exportMemory", exportMemory),
    vscode.commands.registerCommand("waggle.openInstallDocs", openInstallDocs),
    vscode.commands.registerCommand("waggle.queryMemory", queryMemory),
    vscode.commands.registerCommand("waggle.observeConversation", observeConversation)
  );

  void maybePromptInstall();
}
