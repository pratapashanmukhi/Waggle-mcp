import * as vscode from "vscode";
import { ServerManager } from "./server-manager";

export class GraphStudioViewProvider implements vscode.WebviewViewProvider {
  static readonly viewType = "waggle.graphStudio";

  private view: vscode.WebviewView | undefined;

  constructor(private readonly serverManager: ServerManager) {}

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this.view = webviewView;
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: []
    };
    this.refresh();
  }

  refresh(): void {
    if (!this.view) {
      return;
    }
    const runtime = this.serverManager.runtime;
    if (!runtime) {
      this.view.webview.html = `<!DOCTYPE html>
<html><body style="font-family: sans-serif; padding: 1rem;">
<p>Waggle server is not running.</p>
<p>Run <strong>Waggle: Enable for this Workspace</strong> or turn on <code>waggle.autoStart</code>.</p>
</body></html>`;
      return;
    }

    const url = `${runtime.baseUrl}/graph?mode=edit`;
    const origin = runtime.baseUrl;
    const csp = [
      "default-src 'none'",
      `frame-src ${origin}`,
      `img-src ${origin} https: data:`,
      `style-src ${origin} 'unsafe-inline'`,
      `script-src ${origin} 'unsafe-inline'`,
      `connect-src ${origin} ws://127.0.0.1:*`,
      `font-src ${origin} https: data:`
    ].join("; ");

    this.view.webview.html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <meta http-equiv="refresh" content="0; url=${url}">
  <style>body { margin: 0; }</style>
</head>
<body>
  <iframe src="${url}" style="border:0;width:100%;height:100vh;" title="Waggle Graph Studio"></iframe>
</body>
</html>`;
  }
}
