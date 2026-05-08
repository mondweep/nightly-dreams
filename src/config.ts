export interface ScheduledPrompt {
  id: string;
  name: string;
  description: string;
  gistUrl: string;
  schedule: string; // cron format
  enabled: boolean;
}

export interface Config {
  prompts: ScheduledPrompt[];
}

export const defaultConfig: Config = {
  prompts: [
    {
      id: "ruvector-research",
      name: "Ruvector Research Workflow",
      description: "Autonomous research workflow for Rust vector database improvements",
      gistUrl: "https://gist.github.com/ruvnet/1db9a4b9ea9408e880b39a172868496c",
      schedule: "0 0 * * *", // Daily at midnight UTC
      enabled: true
    }
  ]
};
