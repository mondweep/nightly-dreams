import "dotenv/config";
import { PromptScheduler } from "./scheduler.js";
import { defaultConfig } from "./config.js";

async function main() {
  console.log("🌙 Nightly Dreams - Prompt Scheduler");
  console.log("=====================================\n");

  const scheduler = new PromptScheduler();

  // Schedule all configured prompts
  scheduler.schedule(defaultConfig);

  // Execute the first prompt immediately for testing
  if (defaultConfig.prompts.length > 0) {
    console.log("\n📋 Executing initial prompt...");
    await scheduler.executeNow(defaultConfig.prompts[0].id, defaultConfig);
  }

  // Keep the process running
  console.log("\n⏱️  Scheduler is running. Press Ctrl+C to stop.\n");

  // Graceful shutdown
  process.on("SIGINT", () => {
    console.log("\n\nShutting down...");
    scheduler.stop();
    process.exit(0);
  });
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
