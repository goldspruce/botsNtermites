# Copilot / AI Agent Instructions for botsNtermites

Purpose
- Short, targeted guidance so an AI coding agent can be productive immediately.

Quick setup & run
- Install runtime deps: `pip install pygame numpy matplotlib`
- Run interactive Pygame simulations:
  - `python game.py` — default interactive demo (ENTER shows metrics, SPACE restarts)
  - `python game_circle.py`, `python game_circle_corner.py` — alternate scenes
  - `python game_circle_evolve.py` — evolution/evaluation harness (headless + visual replay)
- Run stigmergy / batch animations:
  - `python termites.py` — Matplotlib animation (press ENTER in plot to start)
  - Other variants: `termitesN.py`, `termitesN2.py`, `termites0.py`, `termites1.py`, `termites3.py`

Big picture architecture
- Two simulation styles:
  1. Real-time interactive GUI using Pygame (files: [game.py](game.py), [game_circle.py](game_circle.py), [game_circle_corner.py](game_circle_corner.py)).
  2. Batch/analysis animations using NumPy + Matplotlib (files: [termites.py](termites.py) and variants).
- Common structure across Pygame scripts:
  - Top-level constants (ENV/physics/visuals) — change these to alter scenarios.
  - `Robot` and `Block` classes implementing local state and behavior.
  - Update loop pattern: sense -> decide -> move -> physics (collisions) -> render -> metrics.
  - Metrics & final report functions (e.g., `show_results` / `show_results_screen`) rely on `Block.start_x/start_y` snapshots.

Project-specific conventions & patterns
- Constants-first: parameters are defined at file top; prefer editing constants over inlining values.
- Sense–Decide–Act: implement new behaviors inside `Robot.sense()` and `Robot.decide_and_move()`.
- Raycasting shadows: Pygame scripts use `pygame.Rect.clipline()` to test if a line from `LIGHT_POS` to a sensor point intersects a block — modify this call when changing shadow logic.
- Spawn modes: many files expose `BLOCK_SPAWN_MODE` / `ROBOT_SPAWN_MODE` or bespoke helpers (e.g., `get_block_spawn_pos`, `get_robot_spawn_pos`) — change these to vary initial conditions.
- Metric names:
  - Pygame demos: "peripherality" (distance-from-light / percent) measured per block.
  - Termites demos: "order" metric computed by neighbor counts (see `calculate_order()` in `termites*.py`).

Integration points & dependencies
- Purely local; no network calls. External libs used: `pygame`, `numpy`, `matplotlib`.
- Evolution experiments run headless (visual=False) for fitness evaluation, then replay best genomes visually. See `game_circle_evolve.py` and `termitesN.py` for examples.

Developer workflows & quick notes
- Interactive controls (Pygame): `ENTER` stops sim and shows metrics; `SPACE` restarts.
- Headless/evolution runs: many scripts expose a `visual` flag or separate headless runner functions — run these for fast CI-friendly experiments.
- If display is unavailable (CI/headless macOS), use the Matplotlib headless routines or evolution headless functions rather than Pygame.

Editing guidance
- To add/modify agent behavior: change `Robot.sense()` and `Robot.decide_and_move()` in the relevant script.
- To change physics: edit `check_collisions()` or physics helper functions (often modularized per variant).
- To preserve metrics, ensure `Block.start_x`/`start_y` are set at spawn and not overwritten.

Important files to inspect first
- Primary interactive demo: [game.py](game.py)
- Circular-physics variant: [game_circle.py](game_circle.py)
- Corner/entry scenrio: [game_circle_corner.py](game_circle_corner.py)
- Evolution / GA runner: [game_circle_evolve.py](game_circle_evolve.py)
- Stigmergy / grid-based demos: [termites.py](termites.py) and [termitesN.py](termitesN.py)

When unsure
- Prefer local, minimal edits: tweak top-of-file constants and test.
- For new behaviors, implement inside `Robot` methods and run the corresponding scenario to verify visuals and metrics.

Please review: any unclear areas or missing examples you'd like added?
# Copilot / AI Agent Instructions for botsNtermites

Summary
- Small collection of Python simulation scripts for stigmergy/agent behavior.
- Two main simulation styles: real-time GUI using `pygame` (`game.py` family) and
  batch/animation using `matplotlib` + `numpy` (`termites.py`).

Quick run commands
- Install dependencies: `pip install pygame numpy matplotlib`
- Run interactive swarm demo (Pygame): `python game.py`  # Press ENTER to show metrics, SPACE to restart
- Run stigmergy simulation (Matplotlib): `python termites.py`  # Press ENTER in plot window to start

Big picture (architecture & data flows)
- `game.py` is the orchestrator for the Pygame demo: constants at the top, then
  `Robot` and `Block` classes, physics (`check_collisions`), rendering and a
  `main()` loop. The update pattern is: sense -> decide -> move -> collisions -> draw.
- `game_circle*.py` variants are alternative scenes/experiments that share the
  same top-level approach (constants -> simple classes -> simulation loop).
- `termites.py` implements a grid-based stigmergy model using NumPy arrays and
  a Matplotlib animation loop; metrics (order/clustering) are computed in
  `calculate_order()` and snapshots are gathered for a timeline.

Project-specific conventions and patterns
- Global-constants-first: simulation parameters (grid/size/speeds) live as
  top-level constants in each script — prefer editing those rather than
  hardcoding values in functions.
- Update cycle pattern (common across files):
  1) Sense (read environment) 2) Decide (state update) 3) Act/Move 4) Physics/collisions 5) Render/record
- Metric/report triggers are UI-driven: in `game.py` press ENTER to stop and
  show the results screen; ensure `final_time` logic remains untouched when
  changing termination conditions.
- Procedural single-file layout: scripts are runnable modules with `if __name__ == "__main__": main()`
  — prefer editing the file in place rather than refactoring into packages unless you need reuse.

Integration points & dependencies
- Pygame GUI: `game.py` family. Pay attention to `pygame.Rect.clipline` usage
  for raycasting/shadow checks in `Robot.sense()`.
- NumPy + Matplotlib: `termites.py`. The grid is a NumPy 2D `int` array and
  neighbor calculations use `np.roll` patterns.
- External calls: none to network or services — pure local simulation.

Editing guidance & examples
- To add a new behavior to the Pygame agents, implement it inside `Robot.sense()`
  and/or `Robot.decide_and_move()` and keep changes consistent with the
  `touching_block` and `in_shadow` flags (these interact with collision logic).
- Example: to log per-frame safe robot count, insert a small accumulator in
  the main loop (after robots update) rather than changing `Robot` internals.

Debugging & local testing notes
- Pygame errors: run `python game.py` from a terminal to see tracebacks.
- If display not available (headless), run `termites.py` (non-Pygame) or use
  a virtual frame buffer for Pygame tests (not currently configured).

Files of interest
- `game.py` — main Pygame simulation (Robot/Block + metrics screen)
- `game_circle*.py` — variants/experiments of the Pygame simulation
- `termites.py` — grid-based stigmergy simulation using NumPy/Matplotlib

When you're unsure
- Prefer local, minimal changes: tweak constants at the top, run the script,
  verify visual/metric changes. Open an issue or ask for clarification if you
  plan to refactor multiple files or add CI/tests.

Feedback
- I drafted these instructions from the repo sources. Tell me what’s missing
  or unclear (examples, command flags, or additional files to reference) and
  I will update the file.
