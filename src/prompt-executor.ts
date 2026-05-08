import axios from "axios";
import { ScheduledPrompt } from "./config.js";

export interface ExecutionResult {
  promptId: string;
  timestamp: Date;
  success: boolean;
  gistContent?: string;
  error?: string;
  executedAt: string;
}

export async function fetchGistContent(gistUrl: string): Promise<string> {
  // Extract gist ID from URL
  const gistMatch = gistUrl.match(/\/([a-f0-9]+)$/);
  if (!gistMatch) {
    throw new Error(`Invalid gist URL format: ${gistUrl}`);
  }

  const gistId = gistMatch[1];
  const rawUrl = `https://gist.githubusercontent.com/ruvnet/${gistId}/raw`;

  const response = await axios.get(rawUrl);
  return response.data;
}

export async function executePrompt(
  prompt: ScheduledPrompt
): Promise<ExecutionResult> {
  const timestamp = new Date();
  console.log(
    `[${timestamp.toISOString()}] Executing prompt: ${prompt.name}`
  );

  try {
    const gistContent = await fetchGistContent(prompt.gistUrl);

    console.log(`Successfully fetched gist content (${gistContent.length} chars)`);

    // Log the content for now - in a real implementation,
    // this would call Claude API or trigger the workflow
    console.log(`\n--- Gist Content Preview ---\n${gistContent.substring(0, 500)}...\n`);

    return {
      promptId: prompt.id,
      timestamp,
      success: true,
      gistContent,
      executedAt: timestamp.toISOString()
    };
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : String(error);
    console.error(`Failed to execute prompt ${prompt.id}: ${errorMessage}`);

    return {
      promptId: prompt.id,
      timestamp,
      success: false,
      error: errorMessage,
      executedAt: timestamp.toISOString()
    };
  }
}
