# Philosophy

HarnessGym is built on one observation: **a coding agent is only as good as the
tools in its workspace, and most of those tools don't exist yet.**

A capable agent dropped into a hard kernel-optimization task spends much of its
budget re-deriving instrumentation a human expert would already have on hand —
a way to dump assembly, a rollback-safe variant sweeper, a held-out evaluator,
a place to keep benchmark history. It builds these ad hoc, uses them once, and
throws them away when the session ends. The next session starts from zero.

HarnessGym refuses to throw them away.

## Five principles

### 1. Build the tool once; carry it forward

The unit of progress is not the solved task — it's the **reusable artifact**.
After every attempt, the same session reflects on the single
highest-leverage thing it was missing and builds exactly that. The artifact
lives under `.harnessgym/` and is activated for every future session. Solving
the task is almost a side effect; accumulating capability is the point.

### 2. One artifact per iteration

Reflection picks **one** improvement, not a backlog. A focused MCP server with
real self-tests beats five half-built scripts. Subsequent iterations *extend
and harden* the existing suite rather than scattering disconnected wrappers, so
the harness matures instead of sprawling.

### 3. Fresh attempt, accumulated harness

Every attempt is a new session, because the interesting question is *"how well
does a clean agent do with this tooling?"* — not *"how well does an agent do
after an hour of conversation context."* The conversation resets; the
`.harnessgym/` registry does not. This separation is what makes a replay
comparison meaningful.

### 4. A harness only counts if it works in a clean room

A generated tool that looks helpful but fails to activate is worse than
nothing — it wastes the next session's attention. So nothing is trusted until
it is copied into a **fresh workspace**, activated, and self-tested. Failures
get one or more repair attempts in-session; anything still broken is
**quarantined** — kept for evidence, hidden from every future attempt. The
default is suspicion, not optimism.

### 5. Measure tool *use*, not tool *presence*

It is easy to fool yourself into thinking a harness helped when the agent
merely had it available. HarnessGym logs every generated MCP `tools/call` with
real telemetry, and a harnessed comparison can be configured to count **only**
if the agent actually called a generated tool. Activation is not evidence; a
recorded call is.

## What HarnessGym deliberately is not

- **Not an agent.** HarnessGym never writes your kernel. It orchestrates the
  agent you bring (Codex or Claude Code) and keeps what that agent generates.
- **Not a model harness in the prompt-engineering sense.** It doesn't tune
  system prompts for benchmark scores. It builds *durable tools* and measures
  whether they transfer to a fresh session.
- **Not a leaderboard.** The bundled results are evidence the loop works, run on
  one machine. They are reproducible, not statistically powered.
- **Not magic.** If the agent can't make progress on a task even with good
  tools, HarnessGym will faithfully report that — including the timed-out
  attempts and the quarantined artifacts.

## The bet

The bet is that **most of the gap between a generic coding agent and an expert
on a specialized task is missing instrumentation, not missing intelligence** —
and that instrumentation is exactly the kind of thing an agent can build for
itself if you give it a structured loop, hold it to a clean-room qualification
bar, and let the result accumulate.

[**See how the loop enforces this →**](how-it-works.md){ .md-button }
