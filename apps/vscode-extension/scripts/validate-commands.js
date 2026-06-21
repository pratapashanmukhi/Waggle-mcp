const fs = require("fs");
const path = require("path");

function main() {
  const extensionDir = path.resolve(__dirname, "..");
  const packageJsonPath = path.join(extensionDir, "package.json");
  const extensionTsPath = path.join(extensionDir, "src", "extension.ts");

  console.log("Validating VS Code command contributions vs implementation...");
  console.log(`package.json: ${packageJsonPath}`);
  console.log(`extension.ts: ${extensionTsPath}`);

  if (!fs.existsSync(packageJsonPath)) {
    console.error(`Error: package.json not found at ${packageJsonPath}`);
    process.exit(1);
  }
  if (!fs.existsSync(extensionTsPath)) {
    console.error(`Error: extension.ts not found at ${extensionTsPath}`);
    process.exit(1);
  }

  // 1. Extract commands from package.json
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, "utf8"));
  const contributes = packageJson.contributes || {};
  const contributedCommands = (contributes.commands || []).map(cmd => cmd.command);

  console.log(`Found ${contributedCommands.length} command(s) in package.json:`);
  contributedCommands.forEach(cmd => console.log(`  - ${cmd}`));

  // 2. Extract registered commands from src/extension.ts
  const extensionTsContent = fs.readFileSync(extensionTsPath, "utf8");
  const registeredCommands = [];
  const registerCommandRegex = /vscode\.commands\.registerCommand\s*\(\s*['"]([^'"]+)['"]/g;
  let match;
  while ((match = registerCommandRegex.exec(extensionTsContent)) !== null) {
    registeredCommands.push(match[1]);
  }

  console.log(`Found ${registeredCommands.length} registered command(s) in src/extension.ts:`);
  registeredCommands.forEach(cmd => console.log(`  - ${cmd}`));

  // 3. Validate matching
  let hasErrors = false;

  // Check 3.1: Are all contributed commands registered in extension.ts?
  const missingInCode = contributedCommands.filter(cmd => !registeredCommands.includes(cmd));
  if (missingInCode.length > 0) {
    console.error("\n[ERROR] The following command(s) are declared in package.json but not registered in src/extension.ts:");
    missingInCode.forEach(cmd => console.error(`  - ${cmd}`));
    hasErrors = true;
  }

  // Check 3.2: Are all registered 'waggle.' commands declared in package.json?
  const unregisteredInPackage = registeredCommands.filter(cmd => {
    // Only check 'waggle.' prefixed commands to avoid matching potential external commands
    return cmd.startsWith("waggle.") && !contributedCommands.includes(cmd);
  });
  if (unregisteredInPackage.length > 0) {
    console.error("\n[ERROR] The following command(s) are registered in src/extension.ts but not declared in package.json:");
    unregisteredInPackage.forEach(cmd => console.error(`  - ${cmd}`));
    hasErrors = true;
  }

  if (hasErrors) {
    console.error("\nValidation failed! Please fix the command drift.");
    process.exit(1);
  }

  console.log("\nValidation succeeded! All command contributions are properly registered.");
  process.exit(0);
}

main();
