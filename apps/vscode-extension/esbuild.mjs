import * as esbuild from "esbuild";

await esbuild.build({
  entryPoints: ["src/extension.ts", "src/platform.ts"],
  bundle: true,
  outdir: "dist",
  external: ["vscode"],
  platform: "node",
  format: "cjs",
  target: "node18",
  sourcemap: true,
  minify: false,
  logLevel: "info"
});
