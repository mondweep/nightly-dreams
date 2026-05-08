import schedule from "node-schedule";
import { ScheduledPrompt, Config } from "./config.js";
import { executePrompt, ExecutionResult } from "./prompt-executor.js";

export class PromptScheduler {
  private jobs: Map<string, schedule.Job> = new Map();
  private executionHistory: ExecutionResult[] = [];

  schedule(config: Config): void {
    console.log(`Scheduling ${config.prompts.length} prompts...`);

    for (const prompt of config.prompts) {
      if (!prompt.enabled) {
        console.log(`⏭️  Skipped (disabled): ${prompt.name}`);
        continue;
      }

      try {
        const job = schedule.scheduleJob(prompt.schedule, async () => {
          console.log(`\n🚀 Running scheduled job: ${prompt.name}`);
          const result = await executePrompt(prompt);
          this.executionHistory.push(result);

          if (result.success) {
            console.log(`✅ Completed: ${prompt.name}`);
          } else {
            console.log(`❌ Failed: ${prompt.name} - ${result.error}`);
          }
        });

        this.jobs.set(prompt.id, job);
        console.log(`✓ Scheduled: ${prompt.name} (${prompt.schedule})`);
      } catch (error) {
        console.error(`Failed to schedule ${prompt.name}:`, error);
      }
    }

    console.log(`\n${this.jobs.size} job(s) scheduled successfully`);
  }

  async executeNow(promptId: string, config: Config): Promise<ExecutionResult | null> {
    const prompt = config.prompts.find((p) => p.id === promptId);
    if (!prompt) {
      console.error(`Prompt not found: ${promptId}`);
      return null;
    }

    console.log(`\n⚡ Executing immediately: ${prompt.name}`);
    const result = await executePrompt(prompt);
    this.executionHistory.push(result);
    return result;
  }

  stop(): void {
    console.log("\nStopping scheduler...");
    for (const [id, job] of this.jobs) {
      job.cancel();
      console.log(`Cancelled: ${id}`);
    }
    this.jobs.clear();
  }

  getExecutionHistory(): ExecutionResult[] {
    return this.executionHistory;
  }

  getScheduledJobs(): string[] {
    return Array.from(this.jobs.keys());
  }
}
