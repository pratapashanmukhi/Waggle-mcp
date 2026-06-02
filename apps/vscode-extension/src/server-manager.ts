import { spawn, type ChildProcess } from "child_process";
import * as fs from "fs/promises";
import * as http from "http";
import * as path from "path";
import * as vscode from "vscode";
import { BinaryResolver } from "./binary-resolver";

export interface ServerRuntime {
  baseUrl: string;
  port: number;
  command: string;
}

export class ServerManager {
  private child: ChildProcess | undefined;
  private port = 0;
  private readonly onDidChangeEmitter = new vscode.EventEmitter<ServerRuntime | undefined>();
  readonly onDidChange = this.onDidChangeEmitter.event;

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly resolver: BinaryResolver,
    private readonly log: (message: string) => void
  ) {}

  get runtime(): ServerRuntime | undefined {
    if (!this.port || !this.child) {
      return undefined;
    }
    return {
      baseUrl: `http://127.0.0.1:${this.port}`,
      port: this.port,
      command: ""
    };
  }

  async start(env: Record<string, string>, cwd?: string): Promise<ServerRuntime> {
    if (this.child) {
      return (await this.waitForHealthy(this.port))
        ? { baseUrl: `http://127.0.0.1:${this.port}`, port: this.port, command: "" }
        : this.restart(env, cwd);
    }

    const command = await this.resolver.resolveCommandPath();
    this.port = this.context.globalState.get<number>("waggle.httpPort", 18765);
    const args = ["graph-studio", "--host", "127.0.0.1", "--port", String(this.port), "--no-open"];
    this.log(`Starting ${command} ${args.join(" ")}`);

    const child = spawn(command, args, {
      cwd,
      env: { ...process.env, ...env },
      detached: false,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true
    });
    this.child = child;
    child.stdout?.on("data", (chunk) => this.log(String(chunk).trimEnd()));
    child.stderr?.on("data", (chunk) => this.log(String(chunk).trimEnd()));
    child.on("exit", (code) => {
      this.log(`graph-studio exited (${String(code ?? 0)})`);
      if (this.child === child) {
        this.child = undefined;
        this.onDidChangeEmitter.fire(undefined);
      }
    });

    const healthy = await this.waitForHealthy(this.port, 60_000);
    if (!healthy) {
      await this.stop();
      throw new Error(`Waggle server did not become healthy on port ${this.port}`);
    }

    await this.context.globalState.update("waggle.httpPort", this.port);
    const runtime = { baseUrl: `http://127.0.0.1:${this.port}`, port: this.port, command };
    this.onDidChangeEmitter.fire(runtime);
    return runtime;
  }

  async restart(env: Record<string, string>, cwd?: string): Promise<ServerRuntime> {
    await this.stop();
    return this.start(env, cwd);
  }

  async stop(): Promise<void> {
    const child = this.child;
    this.child = undefined;
    if (!child) {
      return;
    }
    if (process.platform === "win32") {
      spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"], { windowsHide: true });
    } else {
      child.kill("SIGTERM");
    }
    this.onDidChangeEmitter.fire(undefined);
  }

  private async waitForHealthy(port: number, timeoutMs = 30_000): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      if (await this.probe(port)) {
        return true;
      }
      await new Promise((resolve) => setTimeout(resolve, 400));
    }
    return false;
  }

  private probe(port: number): Promise<boolean> {
    return new Promise((resolve) => {
      const request = http.get(`http://127.0.0.1:${port}/health/live`, (response) => {
        resolve(response.statusCode === 200);
        response.resume();
      });
      request.on("error", () => resolve(false));
      request.setTimeout(2000, () => {
        request.destroy();
        resolve(false);
      });
    });
  }

  async readPortFile(): Promise<number | undefined> {
    try {
      const filePath = path.join(this.context.globalStorageUri.fsPath, "waggle-port.json");
      const raw = await fs.readFile(filePath, "utf8");
      const payload = JSON.parse(raw) as { port?: number };
      return typeof payload.port === "number" ? payload.port : undefined;
    } catch {
      return undefined;
    }
  }

  async writePortFile(port: number): Promise<void> {
    const filePath = path.join(this.context.globalStorageUri.fsPath, "waggle-port.json");
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, `${JSON.stringify({ port })}\n`, "utf8");
  }
}
