import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import copy
import os
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
GRID_SIZE = 50
NUM_TERMITES = 100
DENSITY_BRICKS = 0.3
TOTAL_STEPS_VISUAL = 2000   # Shorter for visual demo (change as needed)
STEPS_PER_FRAME = 50        # Speed of visual playback

# GENETIC ALGORITHM SETTINGS
POPULATION_SIZE = 20
GENERATIONS = 15
STEPS_TRAINING = 4000       # Short run for evolution (speed over precision)

# ==========================================
# OUTPUT SETUP
# ==========================================
# Timestamped folder: Day_Hour_Minute
timestamp = datetime.now().strftime("%d_%H_%M")
save_dir = f"run_{timestamp}"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created output directory: {save_dir}/")

# ==========================================
# STATE MACHINE CONSTANTS
# ==========================================
STATE_WAIT_START = 0
STATE_EVOLVING = 1
STATE_WAIT_VISUAL = 2
STATE_VISUAL_RUN = 3
STATE_FINISHED = 4

current_state = STATE_WAIT_START

# ==========================================
# CORE SIMULATION LOGIC (HEADLESS)
# ==========================================
def calculate_order_fast(grid, total_bricks):
    """Fast metric calculation for the training loop"""
    if total_bricks == 0: return 0
    # Shift grid to count neighbors
    neighbors = (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                 np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))
    matches = np.sum(neighbors * grid)
    return matches / (total_bricks * 4)

def run_simulation_headless(genome):
    """
    Runs the simulation WITHOUT graphics to test a genome's fitness.
    Returns: Final Order Metric
    """
    # 1. Setup Environment
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
    grid[random_matrix < DENSITY_BRICKS] = 1
    total_bricks = np.sum(grid)
    
    # 2. Setup Agents
    # Genome mapping:
    # 0-4: Prob to PICK UP given 0-4 neighbors
    # 5-9: Prob to DROP given 0-4 neighbors
    genes_pick = genome[0:5]
    genes_drop = genome[5:10]
    
    agent_x = np.random.randint(0, GRID_SIZE, NUM_TERMITES)
    agent_y = np.random.randint(0, GRID_SIZE, NUM_TERMITES)
    agent_carrying = np.zeros(NUM_TERMITES, dtype=bool)
    
    # 3. Fast Loop
    for _ in range(STEPS_TRAINING):
        # Loop through agents
        for i in range(NUM_TERMITES):
            # Random Walk
            agent_x[i] = (agent_x[i] + random.choice([-1, 0, 1])) % GRID_SIZE
            agent_y[i] = (agent_y[i] + random.choice([-1, 0, 1])) % GRID_SIZE
            
            x, y = agent_x[i], agent_y[i]
            here = grid[x, y]
            
            # Count Neighbors
            n_cnt = 0
            if grid[(x+1)%GRID_SIZE, y] == 1: n_cnt += 1
            if grid[(x-1)%GRID_SIZE, y] == 1: n_cnt += 1
            if grid[x, (y+1)%GRID_SIZE] == 1: n_cnt += 1
            if grid[x, (y-1)%GRID_SIZE] == 1: n_cnt += 1
            
            # GENE EXPRESSION (The Logic)
            if not agent_carrying[i] and here == 1:
                # Check PICK probability for this neighbor count
                if random.random() < genes_pick[n_cnt]:
                    agent_carrying[i] = True
                    grid[x, y] = 0
            
            elif agent_carrying[i] and here == 0:
                # Check DROP probability for this neighbor count
                if random.random() < genes_drop[n_cnt]:
                    agent_carrying[i] = False
                    grid[x, y] = 1
                    
    return calculate_order_fast(grid, total_bricks)

# ==========================================
# EVOLUTION MANAGEMENT CLASS
# ==========================================
class EvolutionManager:
    def __init__(self):
        self.population = [[random.random() for _ in range(10)] for _ in range(POPULATION_SIZE)]
        self.current_gen = 0
        self.best_genome = None
        self.best_score = 0
        self.log_text = [] # To store lines of text for display

    def mutate(self, genome):
        # Mutate one gene slightly
        idx = random.randint(0, 9)
        genome[idx] = max(0.0, min(1.0, genome[idx] + random.uniform(-0.2, 0.2)))
        return genome

    def evolve_step(self):
        """Runs ONE generation. Returns True if finished, False if continuing."""
        if self.current_gen >= GENERATIONS:
            return True

        scores = []
        for i, genome in enumerate(self.population):
            score = run_simulation_headless(genome)
            scores.append((score, genome))
        
        # Sort by fitness (Highest Order Metric is best)
        scores.sort(key=lambda x: x[0], reverse=True)
        self.best_score, self.best_genome = scores[0]
        
        # Log for interface - UPDATED to show full genome and keep history
        # Format genes nicely to 2 decimal places so they fit on screen
        genes_str = "[" + ", ".join([f"{x:.2f}" for x in self.best_genome]) + "]"
        log_line = f"G{self.current_gen+1:02d}|Fit:{self.best_score:.4f}|{genes_str}"
        self.log_text.append(log_line)
        # REMOVED the line that popped old logs so we see everything
        
        # Selection: Keep top 20% (Elitism)
        top_count = int(POPULATION_SIZE * 0.2)
        survivors = [s[1] for s in scores[:top_count]]
        
        # Reproduction
        next_gen = survivors[:]
        while len(next_gen) < POPULATION_SIZE:
            parent = random.choice(survivors)
            child = self.mutate(copy.deepcopy(parent))
            next_gen.append(child)
            
        self.population = next_gen
        self.current_gen += 1
        return False

evo_manager = EvolutionManager()

# ==========================================
# VISUALIZATION & INTERFACE
# ==========================================
# Setup Visuals - Increased width slightly for long genomes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Grid Setup
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
cmap = plt.matplotlib.colors.ListedColormap(['white', 'firebrick', 'cornflowerblue', 'lime'])
bounds = [0, 1, 2, 3, 4]
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
img = ax1.imshow(grid, interpolation='nearest', cmap=cmap, norm=norm)
ax1.axis('off')

# Graph Setup
line, = ax2.plot([], [], lw=2, color='firebrick')
ax2.set_xlim(0, TOTAL_STEPS_VISUAL)
ax2.set_ylim(0.0, 1.0)
ax2.set_title("Metric History")
ax2.grid(True)

# Visual Simulation State Variables
visual_grid = None
visual_agents = []
visual_step = 0
history_metric = []
history_steps = []
final_screenshot_taken = False

class Termite:
    def __init__(self):
        self.x = random.randint(0, GRID_SIZE-1)
        self.y = random.randint(0, GRID_SIZE-1)
        self.carrying = False

def init_visual_run():
    global visual_grid, visual_agents, visual_step, history_metric, history_steps
    visual_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
    visual_grid[random_matrix < DENSITY_BRICKS] = 1
    visual_agents = [Termite() for _ in range(NUM_TERMITES)]
    visual_step = 0
    history_metric = []
    history_steps = []
    
    # Save "Start" Screenshot
    ax1.set_title("RUNNING FITTEST SPECIES (START)")
    # Redraw to ensure title is caught
    fig.canvas.draw()
    path = os.path.join(save_dir, "02_fittest_start.png")
    plt.savefig(path)
    print(f"Saved: {path}")

# ==========================================
# MAIN LOOP
# ==========================================
def update(frame):
    global current_state, visual_step, final_screenshot_taken

    # --- STATE 0: WAIT FOR START ---
    if current_state == STATE_WAIT_START:
        ax1.clear()
        ax1.axis('off')
        ax1.text(0.5, 0.5, "PRESS ENTER\nTO START EVOLUTION", 
                 horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=16)
        return []

    # --- STATE 1: EVOLVING ---
    if current_state == STATE_EVOLVING:
        ax1.clear()
        ax1.axis('off')
        ax1.set_title("🧬 EVOLUTION IN PROGRESS...")
        
        # Run one generation
        finished = evo_manager.evolve_step()
        
        # Update Log Text on Screen
        log_str = "\n".join(evo_manager.log_text)
        # UPDATED: Align text to top-left and use smaller font to fit all gens
        ax1.text(0.02, 0.98, log_str, transform=ax1.transAxes, fontsize=9, 
                 fontfamily='monospace', verticalalignment='top')
        
        if finished:
            current_state = STATE_WAIT_VISUAL
            # Save "Evolution Log" Screenshot
            # Redraw needed to ensure latest text renders before save
            fig.canvas.draw()
            path = os.path.join(save_dir, "01_evolution_log.png")
            plt.savefig(path)
            print(f"Saved: {path}")
            
        return []

    # --- STATE 2: WAIT FOR VISUAL ---
    if current_state == STATE_WAIT_VISUAL:
        ax1.clear()
        ax1.axis('off')
        msg = f"EVOLUTION COMPLETE\nBest Score: {evo_manager.best_score:.3f}\n\nPRESS ENTER\nTO RUN CHAMPION"
        ax1.text(0.5, 0.5, msg, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=14)
        return []

    # --- STATE 3: VISUAL RUN ---
    if current_state == STATE_VISUAL_RUN:
        ax1.clear()
        ax1.axis('off')
        ax1.set_title(f"Running Champion | Step {visual_step}")

        # Extract Genes
        genes_pick = evo_manager.best_genome[0:5]
        genes_drop = evo_manager.best_genome[5:10]

        for _ in range(STEPS_PER_FRAME):
            if visual_step >= TOTAL_STEPS_VISUAL: break
            visual_step += 1
            
            for t in visual_agents:
                # Move
                t.x = (t.x + random.choice([-1, 0, 1])) % GRID_SIZE
                t.y = (t.y + random.choice([-1, 0, 1])) % GRID_SIZE
                x, y = t.x, t.y
                here = visual_grid[x, y]
                
                # Neighbors
                n_cnt = 0
                if visual_grid[(x+1)%GRID_SIZE, y] == 1: n_cnt += 1
                if visual_grid[(x-1)%GRID_SIZE, y] == 1: n_cnt += 1
                if visual_grid[x, (y+1)%GRID_SIZE] == 1: n_cnt += 1
                if visual_grid[x, (y-1)%GRID_SIZE] == 1: n_cnt += 1
                
                # Apply Genes
                if not t.carrying and here == 1:
                    if random.random() < genes_pick[n_cnt]:
                        t.carrying = True
                        visual_grid[x, y] = 0
                elif t.carrying and here == 0:
                    if random.random() < genes_drop[n_cnt]:
                        t.carrying = False
                        visual_grid[x, y] = 1

        # Draw Grid
        display = visual_grid.copy()
        for t in visual_agents: display[t.x, t.y] = 3 if t.carrying else 2
        ax1.imshow(display, interpolation='nearest', cmap=cmap, norm=norm)
        
        # Update Graph
        total_bricks = np.sum(visual_grid == 1) # Approximation for order metric calc
        metric = calculate_order_fast(visual_grid, total_bricks) # Note: Total bricks changes slightly as agents pick up
        history_steps.append(visual_step)
        history_metric.append(metric)
        line.set_data(history_steps, history_metric)
        ax2.relim()
        ax2.autoscale_view()

        if visual_step >= TOTAL_STEPS_VISUAL:
            if not final_screenshot_taken:
                ax1.set_title("SIMULATION FINISHED")
                # Redraw to capture updated title/grid state
                fig.canvas.draw()
                path = os.path.join(save_dir, "03_fittest_end.png")
                plt.savefig(path)
                print(f"Saved: {path}")
                final_screenshot_taken = True
            current_state = STATE_FINISHED

        return [img, line]

    return []

# ==========================================
# KEYBOARD INPUT
# ==========================================
def on_key(event):
    global current_state
    if event.key == 'enter':
        if current_state == STATE_WAIT_START:
            current_state = STATE_EVOLVING
        elif current_state == STATE_WAIT_VISUAL:
            init_visual_run()
            current_state = STATE_VISUAL_RUN

fig.canvas.mpl_connect('key_press_event', on_key)

if __name__ == "__main__":
    print("-" * 40)
    print("EVOLUTIONARY TERMITE SIMULATION")
    print(f"Output folder: {save_dir}")
    print("Please use the Graphical Window to control the simulation.")
    print("-" * 40)
    
    # Increased interval slightly to allow graphical updates between evolution steps
    ani = animation.FuncAnimation(fig, update, interval=100, blit=False)
    plt.show()