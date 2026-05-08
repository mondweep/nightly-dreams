# Nightly Dreams - Claude Code Routine

## Purpose
Scheduled execution of the Ruvector research workflow - an autonomous research routine for state-of-the-art vector database improvements in Rust.

## Routine Configuration

**Name:** Ruvector Nightly Research

**Schedule:** Daily at midnight UTC (or customize as needed)

**Prompt:** The routine should fetch and execute the research workflow defined at:
https://gist.github.com/ruvnet/1db9a4b9ea9408e880b39a172868496c

## What the Routine Does

The Ruvector Research Workflow:
1. Reviews project conventions, prior research, and SOTA discoveries
2. Creates a feature branch for research artifacts
3. Generates:
   - Research document with SOTA survey and proposed design
   - Architecture Decision Record (ADR)
   - Working Rust proof-of-concept with real benchmarks
   - Conventional commits
4. Pushes a PR with the research
5. Publishes a public, SEO-optimized GitHub Gist with results
6. Reports final deliverables and metrics

## Key Constraints
- Rust implementation only (no Python/JavaScript/TypeScript)
- Real cargo benchmarks (no placeholder results)
- Code files ≤500 lines each
- No secrets in repositories
- Multiple measured variants (baseline + 2+ alternatives)
- All tests and builds must pass

## How to Create This Routine in Claude Code

1. Open Claude Code
2. Click on the **Routines** section in the sidebar
3. Click **Create Routine** or use `/routine create`
4. Set the following:
   - **Name:** Ruvector Nightly Research
   - **Schedule:** Daily at 00:00 UTC (or your preferred time)
   - **Prompt:** Copy the research workflow from the gist above and set it as the routine prompt
5. Save and activate the routine

The routine will now appear in your Routines section and execute automatically on the schedule.
