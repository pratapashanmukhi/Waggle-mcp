import { defineConfig } from "@vscode/test-cli";

export default defineConfig({
  files: "dist/test/**/*.test.js",
  version: "stable",
  workspaceFolder: "./test-workspace",
  mocha: {
    ui: "bdd"
  }
});
