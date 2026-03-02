#!/usr/bin/env node
const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const BASE_DIR = path.resolve(__dirname, "..");
const REPORT_DIR = path.join(BASE_DIR, "benchmarks");
const REPORT_FILE = path.join(REPORT_DIR, "report.md");
const BENCH_CMD = ["cargo", "bench", "--bench", "boids_update"];

function runBench() {
  const result = spawnSync(BENCH_CMD[0], BENCH_CMD.slice(1), {
    cwd: BASE_DIR,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    process.stdout.write(result.stdout);
    process.stderr.write(result.stderr);
    process.exit(result.status);
  }
  return `${result.stdout || ""}${result.stderr || ""}`.trim();
}

function collectEstimates() {
  const criterionRoot = path.join(BASE_DIR, "target", "criterion");
  const estimates = [];
  if (!fs.existsSync(criterionRoot)) {
    return estimates;
  }
  const stack = [criterionRoot];
  while (stack.length) {
    const current = stack.pop();
    const entries = fs.readdirSync(current, { withFileTypes: true });
    for (const entry of entries) {
      const entryPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(entryPath);
        continue;
      }
      if (entry.isFile() && entry.name === "estimates.json") {
        try {
          const payload = JSON.parse(fs.readFileSync(entryPath, "utf8"));
          const benchId = payload.id;
          const mean = payload.estimates?.mean || {};
          const point = mean.point_estimate;
          const ci = mean.confidence_interval || [];
          if (!benchId || point == null) {
            continue;
          }
          estimates.push({
            id: benchId,
            point,
            ciLow: ci[0] ?? 0,
            ciHigh: ci[1] ?? 0,
          });
        } catch (error) {
          // ignore corrupt records
        }
      }
    }
  }
  return estimates;
}

function formatDuration(value) {
  if (value >= 1) {
    return `${value.toFixed(3)}s`;
  }
  if (value >= 1e-3) {
    return `${(value * 1e3).toFixed(3)}ms`;
  }
  return `${(value * 1e6).toFixed(3)}µs`;
}

function ensureReportFile() {
  fs.mkdirSync(REPORT_DIR, { recursive: true });
  if (!fs.existsSync(REPORT_FILE)) {
    fs.writeFileSync(REPORT_FILE, "# Benchmark Report\n\n");
  }
}

function appendReport(output, estimates) {
  ensureReportFile();
  const lines = output.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const timestamp = new Date().toISOString();
  const header = `## ${timestamp}`;
  const commandLine = BENCH_CMD.join(" ");
  const rows = [];
  rows.push(header);
  rows.push(`- Command: \`${commandLine}\``);
  if (lines.length) {
    rows.push(`- Output:`);
    for (const line of lines) {
      rows.push(`  - ${line}`);
    }
  } else {
    rows.push(`- Output: *(no console output)*`);
  }
  if (estimates.length) {
    rows.push(`- Criterion summaries:`);
    for (const estimate of estimates) {
      rows.push(
        `  - \`${estimate.id}\` — ${formatDuration(estimate.point)} ± [${formatDuration(estimate.ciLow)}, ${formatDuration(estimate.ciHigh)}]`
      );
    }
  }
  fs.appendFileSync(REPORT_FILE, rows.join("\n") + "\n\n");
}

function main() {
  const output = runBench();
  const estimates = collectEstimates();
  appendReport(output, estimates);
}

if (require.main === module) {
  main();
}
