import * as fs from "fs/promises";
import { readFileSync } from "fs";
import * as https from "https";
import * as path from "path";
import * as vscode from "vscode";
import { assetFileNameForCurrentPlatform, platformAssetKey } from "./platform";

export interface BundleMetadata {
  version: string;
  repository: string;
  assets: Record<string, string>;
}

const DEFAULT_REPO = "Abhigyan-Shekhar/Waggle-mcp";

export class BinaryResolver {
  constructor(private readonly context: vscode.ExtensionContext) {}

  private config(): vscode.WorkspaceConfiguration {
    return vscode.workspace.getConfiguration("waggle");
  }

  private cacheRoot(): string {
    return path.join(this.context.globalStorageUri.fsPath, "waggle-bin");
  }

  async resolveCommandPath(): Promise<string> {
    const method = this.config().get<string>("installMethod", "binary");
    if (method === "pipx") {
      return this.config().get<string>("commandPath", "waggle-mcp");
    }
    return this.ensureBinary();
  }

  async hasCachedBinary(): Promise<boolean> {
    try {
      await fs.access(await this.cachedBinaryPath());
      return true;
    } catch {
      return false;
    }
  }

  private async cachedBinaryPath(): Promise<string> {
    const version = await this.resolveBinaryVersion();
    const assetName = assetFileNameForCurrentPlatform();
    return path.join(this.cacheRoot(), version, assetName);
  }

  async ensureBinary(): Promise<string> {
    const destPath = await this.cachedBinaryPath();
    try {
      await fs.access(destPath);
      return destPath;
    } catch {
      // download below
    }

    const version = await this.resolveBinaryVersion();
    const assetName = assetFileNameForCurrentPlatform();
    const destDir = path.dirname(destPath);

    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Waggle",
        cancellable: false
      },
      async () => {
        await fs.mkdir(destDir, { recursive: true });
        const url = await this.resolveDownloadUrl(version, assetName);
        await this.downloadFile(url, destPath);
        if (process.platform !== "win32") {
          await fs.chmod(destPath, 0o755);
        }
      }
    );

    return destPath;
  }

  private async resolveBinaryVersion(): Promise<string> {
    const override = this.config().get<string>("binaryVersion", "").trim();
    if (override) {
      return override;
    }
    return this.context.extension.packageJSON.version as string;
  }

  private async resolveDownloadUrl(version: string, assetName: string): Promise<string> {
    const repo = this.config().get<string>("binaryReleaseRepo", DEFAULT_REPO).trim() || DEFAULT_REPO;
    const metadata = await this.fetchBundleMetadata(repo, version);
    const mapped = metadata.assets[platformAssetKey()];
    if (mapped !== assetName) {
      throw new Error(`Release ${version} asset mismatch for ${platformAssetKey()}`);
    }
    return `https://github.com/${repo}/releases/download/v${metadata.version}/${assetName}`;
  }

  private async fetchBundleMetadata(repo: string, version: string): Promise<BundleMetadata> {
    const fromRelease = await this.tryFetchReleaseMetadata(repo, version);
    if (fromRelease) {
      return fromRelease;
    }
    const latest = await this.tryFetchLatestMetadata(repo);
    if (latest) {
      return latest;
    }
    return this.loadBundledMetadata();
  }

  private async tryFetchReleaseMetadata(repo: string, version: string): Promise<BundleMetadata | undefined> {
    try {
      const release = await this.fetchJson<{ tag_name: string; assets: { name: string; browser_download_url: string }[] }>(
        `https://api.github.com/repos/${repo}/releases/tags/v${version}`
      );
      return this.metadataFromRelease(release);
    } catch {
      return undefined;
    }
  }

  private async tryFetchLatestMetadata(repo: string): Promise<BundleMetadata | undefined> {
    try {
      const release = await this.fetchJson<{ tag_name: string; assets: { name: string; browser_download_url: string }[] }>(
        `https://api.github.com/repos/${repo}/releases/latest`
      );
      return this.metadataFromRelease(release);
    } catch {
      return undefined;
    }
  }

  private metadataFromRelease(release: {
    tag_name: string;
    assets: { name: string }[];
  }): BundleMetadata {
    const version = release.tag_name.replace(/^v/, "");
    const bundled = this.loadBundledMetadataSync();
    const assets: Record<string, string> = { ...bundled.assets };
    for (const asset of release.assets) {
      for (const [key, expected] of Object.entries(assets)) {
        if (asset.name === expected) {
          assets[key] = expected;
        }
      }
    }
    return { version, repository: DEFAULT_REPO, assets };
  }

  private async loadBundledMetadata(): Promise<BundleMetadata> {
    const uri = vscode.Uri.joinPath(this.context.extensionUri, "resources", "bundle-metadata.json");
    const raw = await fs.readFile(uri.fsPath, "utf8");
    return JSON.parse(raw) as BundleMetadata;
  }

  private loadBundledMetadataSync(): BundleMetadata {
    const filePath = path.join(this.context.extensionPath, "resources", "bundle-metadata.json");
    return JSON.parse(readFileSync(filePath, "utf8")) as BundleMetadata;
  }

  private fetchJson<T>(url: string): Promise<T> {
    return new Promise((resolve, reject) => {
      const headers: Record<string, string> = { Accept: "application/vnd.github+json", "User-Agent": "waggle-vscode-extension" };
      const request = https.get(url, { headers }, (response) => {
        if (response.statusCode === 302 || response.statusCode === 301) {
          const location = response.headers.location;
          if (location) {
            void this.fetchJson<T>(location).then(resolve, reject);
            return;
          }
        }
        if ((response.statusCode ?? 0) >= 400) {
          reject(new Error(`GitHub API ${response.statusCode ?? 0} for ${url}`));
          return;
        }
        const chunks: Buffer[] = [];
        response.on("data", (chunk: Buffer) => chunks.push(chunk));
        response.on("end", () => {
          try {
            resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")) as T);
          } catch (error) {
            reject(error);
          }
        });
      });
      request.on("error", reject);
    });
  }

  private downloadFile(url: string, destPath: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = https.get(url, { headers: { "User-Agent": "waggle-vscode-extension" } }, (response) => {
        if (response.statusCode === 302 || response.statusCode === 301) {
          const location = response.headers.location;
          if (!location) {
            reject(new Error(`Redirect without location for ${url}`));
            return;
          }
          void this.downloadFile(location, destPath).then(resolve, reject);
          return;
        }
        if ((response.statusCode ?? 0) >= 400) {
          reject(new Error(`Download failed (${response.statusCode ?? 0}) for ${url}`));
          return;
        }
        const chunks: Buffer[] = [];
        response.on("data", (chunk: Buffer) => chunks.push(chunk));
        response.on("end", () => {
          void fs.writeFile(destPath, Buffer.concat(chunks)).then(() => resolve(), reject);
        });
      });
      request.on("error", reject);
    });
  }
}
