# Nightly Dreams

A scheduling system for executing Claude-based research and development workflows on a recurring basis.

## Overview

Nightly Dreams enables autonomous, scheduled execution of research prompts and workflows. It fetches prompt definitions from GitHub Gists and executes them according to a configurable schedule.

## Current Workflow

The system is configured to execute the **Ruvector Research Workflow**, which is an autonomous research routine for the ruvector Rust vector database project. This workflow:

1. **Reviews** project conventions and state-of-the-art discoveries
2. **Creates** a feature branch with research artifacts
3. **Generates** documentation including:
   - Research documents with SOTA survey
   - Architecture Decision Records (ADRs)
   - Working Rust proof-of-concept with benchmarks
4. **Publishes** results via PR and GitHub Gist
5. **Reports** final deliverables and metrics

### Key Constraints

- Rust-only implementations (no Python/JavaScript/TypeScript)
- Real cargo benchmarks (no placeholder results)
- Code files ≤500 lines
- No secrets in repositories
- Multiple measured variants (baseline + 2+ alternatives)
- All tests and builds must pass

## Installation

```bash
npm install
```

## Configuration

1. Copy `.env.example` to `.env` and configure your API keys:
   ```bash
   cp .env.example .env
   ```

2. Edit `src/config.ts` to modify scheduled prompts:
   - `id`: Unique identifier
   - `name`: Display name
   - `gistUrl`: URL to the prompt gist
   - `schedule`: Cron expression (see [crontab.guru](https://crontab.guru))
   - `enabled`: Whether this prompt should run

## Development

```bash
# Development mode with hot reload
npm run dev

# Build TypeScript
npm build

# Run tests
npm test

# Start scheduler
npm start
```

## Scheduling Format

Prompts use standard cron syntax:

```
 ┌───────────── minute (0 - 59)
 │ ┌───────────── hour (0 - 23)
 │ │ ┌───────────── day of month (1 - 31)
 │ │ │ ┌───────────── month (1 - 12)
 │ │ │ │ ┌───────────── day of week (0 - 6) (Sunday to Saturday)
 │ │ │ │ │
 │ │ │ │ │
 * * * * *
```

Common examples:
- `0 0 * * *` - Every day at midnight
- `0 2 * * 0` - Every Sunday at 2 AM
- `0 9-17 * * 1-5` - Every weekday at 9 AM-5 PM (hourly)

## Project Structure

```
src/
├── index.ts           # Main entry point
├── config.ts          # Configuration and interfaces
├── scheduler.ts       # Schedule management
└── prompt-executor.ts # Prompt fetching and execution
```

## Architecture

1. **Scheduler**: Manages cron jobs using `node-schedule`
2. **Executor**: Fetches gist content and prepares for execution
3. **Config**: Centralized prompt configuration

Future enhancements will integrate with the Claude API to actually execute the prompts and handle the research workflow outputs.

## License

MIT
