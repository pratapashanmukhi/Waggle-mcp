import { execFile, type ExecFileOptions } from "child_process";

export interface CommandResult {
  code: number;
  stdout: string;
  stderr: string;
}

export async function execFileAsync(
  command: string,
  args: string[],
  options?: ExecFileOptions
): Promise<CommandResult> {
  return await new Promise((resolve, reject) => {
    execFile(command, args, { windowsHide: true, ...options }, (error, stdout, stderr) => {
      const numericCode = (error as NodeJS.ErrnoException | null)?.code;
      const code = typeof numericCode === "number" ? numericCode : 0;
      if (error && typeof (error as NodeJS.ErrnoException).code !== "number") {
        reject(error);
        return;
      }
      resolve({ code, stdout: String(stdout), stderr: String(stderr) });
    });
  });
}
