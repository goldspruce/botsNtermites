import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import random
import copy
import os

# ==========================================
# 1. THE LABORATORY SETTINGS (Configuration)
# ==========================================
# The physical world settings
GRID_SIZE = 50          # The world is a 50x50 grid
NUM_TERMITES = 100      # 100 agents working simultaneously
DENSITY_BRICKS = 0.3    # 30% of the world is filled with bricks

# THE EVOLUTIONARY ENGINE
# This is where we simulate "Natural Selection"
POPULATION_SIZE = 20    # We test 20 different "species" (genomes) at a time
GENERATIONS = 20        # We repeat the cycle of "Test -> Kill -> Mutate" 20 times
STEPS_TRAINING = 4000   # <--- RESTORED: Each species gets 4000 moves to prove itself
TRIALS_PER_GENOME = 5   # We test each species 5 times and average the score (To remove luck)

# THE VISUALIZATION
# These settings control the "Movie" you see at the end
TOTAL_STEPS_VISUAL = 10000 
STEPS_PER_FRAME = 200      # How many moves happen between screen updates?
SNAPSHOT_STEPS = [0, 2500, 5000, 7500, 10000] # When to take photos for the gallery

# ==========================================
# 2. THE STATE MACHINE
# ==========================================
# The program works in phases. These numbers tell the computer which phase we are in.
STATE_EVOLVING = 0     # Phase 1: Running invisible math to find the best genes
STATE_TRANSITION = 1   # Phase 2: Setting up the screen for the final show
STATE_SIMULATING = 2   # Phase 3: Showing the best termite in action
STATE_GALLERY = 4      # Phase 4: Showing the photo gallery at the end

current_state = STATE_EVOLVING
current_gen = 0
best_genome_overall = None
best_score_overall = 0.0
saved_snapshots = [] 

# ==========================================
# 3. THE "BRAIN" (Mechanism)
# ==========================================
def calculate_order_fast(grid, total_bricks):
    """
    THE EXAM: This function grades the termites.
    It calculates the 'Order Metric' (0.0 to 1.0).
    Higher Score = More Clumping = Better Architecture.
    """
    if total_bricks == 0: return 0
    # Math trick: Shift the grid up/down/left/right to count neighbors instantly
    neighbors = (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                 np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))
    matches = np.sum(neighbors * grid)
    # Score = (Touching Sides) / (Total Possible Sides)
    return matches / (total_bricks * 4)

def run_simulation_headless(genome):
    """
    THE TEST TRACK: Runs a simulation WITHOUT graphics.
    This happens thousands of times in the background to evolve the species.
    """
    # Create a fresh random world
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
    grid[random_matrix < DENSITY_BRICKS] = 1
    total_bricks = np.sum(grid)
    
    # UNPACK THE GENES (Type I Mechanism)
    # The genome is just a list of 10 numbers.
    # It acts as a look-up table, not a brain.
    genes_pick = genome[0:5] # Probability to PICK UP (if 0, 1, 2, 3, 4 neighbors)
    genes_drop = genome[5:10] # Probability to DROP (if 0, 1, 2, 3, 4 neighbors)
    
    # Place agents randomly
    agent_x = np.random.randint(0, GRID_SIZE, NUM_TERMITES)
    agent_y = np.random.randint(0, GRID_SIZE, NUM_TERMITES)
    agent_carrying = np.zeros(NUM_TERMITES, dtype=bool)
    
    # The Time Loop
    for _ in range(STEPS_TRAINING):
        # 1. MOVE: Random Walk (Brownian Motion)
        shift_x = np.random.choice([-1, 0, 1], NUM_TERMITES)
        shift_y = np.random.choice([-1, 0, 1], NUM_TERMITES)
        agent_x = (agent_x + shift_x) % GRID_SIZE
        agent_y = (agent_y + shift_y) % GRID_SIZE
        
        # 2. SENSE & ACT
        for i in range(NUM_TERMITES):
            x, y = agent_x[i], agent_y[i]
            here = grid[x, y]
            
            # Count neighbors at current location (0 to 4)
            n_cnt = 0
            if grid[(x+1)%GRID_SIZE, y] == 1: n_cnt += 1
            if grid[(x-1)%GRID_SIZE, y] == 1: n_cnt += 1
            if grid[x, (y+1)%GRID_SIZE] == 1: n_cnt += 1
            if grid[x, (y-1)%GRID_SIZE] == 1: n_cnt += 1
            
            # DECISION TREE (Driven by Genes)
            if not agent_carrying[i] and here == 1:
                # If standing on a brick, check the "Pick Up" gene for this neighbor count
                if random.random() < genes_pick[n_cnt]:
                    agent_carrying[i] = True
                    grid[x, y] = 0
            elif agent_carrying[i] and here == 0:
                # If holding a brick on empty ground, check the "Drop" gene
                if random.random() < genes_drop[n_cnt]:
                    agent_carrying[i] = False
                    grid[x, y] = 1
                    
    # Return the final grade for this student
    return calculate_order_fast(grid, total_bricks)

# ==========================================
# 4. EVOLUTION TOOLS
# ==========================================
def create_genome():
    # Create a random "DNA" string of 10 probabilities
    return [random.random() for _ in range(10)]

def mutate(genome):
    """
    THE INNOVATION ENGINE:
    Takes a genome and changes ONE number slightly.
    This mimics biological mutation.
    """
    idx = random.randint(0, 9) # Pick one of the 10 genes
    # Change it by -0.2 to +0.2
    genome[idx] = max(0.0, min(1.0, genome[idx] + random.uniform(-0.2, 0.2)))
    return genome

# Create the first class of 20 random students
population = [create_genome() for _ in range(POPULATION_SIZE)]
evolution_history = [] 

# ==========================================
# 5. GRAPHICS SETUP
# ==========================================
fig = plt.figure(figsize=(12, 7))

# PHASE 1 SCREEN: Evolution Dashboard
ax_evol_text = fig.add_subplot(1, 2, 1)
ax_evol_graph = fig.add_subplot(1, 2, 2)

ax_evol_text.axis('off')
ax_evol_text.set_title("Evolutionary Laboratory (Termites)", fontsize=14, fontweight='bold')
text_display = ax_evol_text.text(0.1, 0.5, "Initializing Population...", fontsize=11, va='center', fontfamily='monospace')

ax_evol_graph.set_title("Fitness Over Generations")
ax_evol_graph.set_xlabel("Generation")
ax_evol_graph.set_ylabel("Order Metric (Avg)")
ax_evol_graph.set_xlim(0, GENERATIONS)
ax_evol_graph.set_ylim(0.5, 0.8) # Zoomed in to show relevant growth
ax_evol_graph.grid(True)
line_fitness, = ax_evol_graph.plot([], [], 'b-o', lw=2)

# Placeholders for PHASE 2 (Simulation)
ax_sim_grid = None
ax_sim_graph = None
img = None
line_sim = None
grid_sim = None
agents = []
sim_step = 0
sim_history_steps = []
sim_history_metric = []

# ==========================================
# 6. MAIN PROGRAM LOOPS
# ==========================================

def start_simulation_phase():
    """
    Triggered when evolution finishes. 
    Wipes the screen and sets up the Visual Demo.
    """
    fig.clf() 
    global ax_sim_grid, ax_sim_graph, img, line_sim, grid_sim, agents 
    global sim_history_steps, sim_history_metric
    
    ax_sim_grid = fig.add_subplot(1, 2, 1)
    ax_sim_graph = fig.add_subplot(1, 2, 2)
    
    # Init Simulation Grid
    grid_sim = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
    grid_sim[random_matrix < DENSITY_BRICKS] = 1
    total_bricks = np.sum(grid_sim)
    
    start_score = calculate_order_fast(grid_sim, total_bricks)
    sim_history_steps = [0]
    sim_history_metric = [start_score]
    
    # Init Agents
    class Termite:
        def __init__(self):
            self.x = random.randint(0, GRID_SIZE-1)
            self.y = random.randint(0, GRID_SIZE-1)
            self.carrying = False
    agents = [Termite() for _ in range(NUM_TERMITES)]
    
    # Visual Styling
    cmap = plt.matplotlib.colors.ListedColormap(['white', 'firebrick', 'cornflowerblue', 'lime'])
    bounds = [0, 1, 2, 3, 4]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    img = ax_sim_grid.imshow(grid_sim, interpolation='nearest', cmap=cmap, norm=norm)
    ax_sim_grid.set_title(f"Termite Visualization (Score: {best_score_overall:.3f})", fontsize=12)
    ax_sim_grid.axis('off')
    
    line_sim, = ax_sim_graph.plot(sim_history_steps, sim_history_metric, lw=2, color='firebrick')
    ax_sim_graph.set_xlim(0, TOTAL_STEPS_VISUAL)
    ax_sim_graph.set_ylim(0.5, 0.85) 
    ax_sim_graph.set_title("Order Metric (Real-time)")
    ax_sim_graph.set_xlabel("Time (Steps)")
    ax_sim_graph.grid(True)

def show_gallery_phase():
    """
    Phase 4: Pop up the gallery of snapshots.
    """
    fig.clf()
    fig.suptitle("Evolutionary Progression: Termite Construction", fontsize=16)
    
    num_snaps = len(saved_snapshots)
    for i, filename in enumerate(saved_snapshots):
        # Create small sub-windows for each image
        ax = fig.add_subplot(1, num_snaps, i+1)
        image = mpimg.imread(filename)
        ax.imshow(image)
        ax.set_title(f"Step {SNAPSHOT_STEPS[i]}")
        ax.axis('off')
        
    plt.draw()

def update(frame):
    global current_state, current_gen, population, best_genome_overall, best_score_overall, sim_step
    
    # -----------------------------
    # PHASE 1: EVOLUTION (The Search)
    # -----------------------------
    if current_state == STATE_EVOLVING:
        if current_gen < GENERATIONS:
            scores = []
            
            # --- MONTE CARLO TESTING ---
            # Test every genome 5 times and average the result
            for i, genome in enumerate(population):
                trial_scores = []
                for _ in range(TRIALS_PER_GENOME): 
                    trial_scores.append(run_simulation_headless(genome))
                avg_score = sum(trial_scores) / len(trial_scores)
                scores.append((avg_score, genome))

            # Rank the students
            scores.sort(key=lambda x: x[0], reverse=True)
            best_gen_score, best_gen_genome = scores[0]
            
            # Remember the "All-Time Champion"
            if best_gen_score > best_score_overall:
                best_score_overall = best_gen_score
                best_genome_overall = best_gen_genome
            
            evolution_history.append(best_gen_score)
            
            # Update Text on Screen
            genes_pick = [round(x, 2) for x in best_gen_genome[0:5]]
            genes_drop = [round(x, 2) for x in best_gen_genome[5:10]]
            
            status_text = (
                f"GENERATION: {current_gen + 1} / {GENERATIONS}\n"
                f"(Avg of {TRIALS_PER_GENOME} trials)\n\n"
                f"🏆 GEN BEST: {best_gen_score:.4f}\n"
                f"👑 ALL-TIME: {best_score_overall:.4f}\n\n"
                f"GENOME (Probabilities):\n"
                f"Pick: {genes_pick}\n"
                f"Drop: {genes_drop}\n"
            )
            text_display.set_text(status_text)
            
            # Update Graph
            gens = range(1, len(evolution_history) + 1)
            line_fitness.set_data(gens, evolution_history)
            
            # --- SURVIVAL OF THE FITTEST ---
            # 1. Keep the top 20% (Elite)
            top_count = int(POPULATION_SIZE * 0.2)
            survivors = [s[1] for s in scores[:top_count]]
            
            # 2. Fill the rest with mutated clones of the survivors
            next_gen = survivors[:]
            while len(next_gen) < POPULATION_SIZE:
                parent = random.choice(survivors)
                # Clone and Mutate
                child = mutate(copy.deepcopy(parent))
                next_gen.append(child)
            population = next_gen
            
            current_gen += 1
            return [line_fitness, text_display]
        else:
            current_state = STATE_TRANSITION
            return []

    # -----------------------------
    # PHASE 2: TRANSITION
    # -----------------------------
    elif current_state == STATE_TRANSITION:
        start_simulation_phase()
        current_state = STATE_SIMULATING
        return [img, line_sim]

    # -----------------------------
    # PHASE 3: SIMULATION (The Demo)
    # -----------------------------
    elif current_state == STATE_SIMULATING:
        if sim_step >= TOTAL_STEPS_VISUAL:
            current_state = STATE_GALLERY
            return []
            
        genes_pick = best_genome_overall[0:5]
        genes_drop = best_genome_overall[5:10]
        
        # Fast-forward the simulation for the visual frame
        for _ in range(STEPS_PER_FRAME):
            if sim_step > TOTAL_STEPS_VISUAL: break
            
            # Take Snapshot?
            if sim_step in SNAPSHOT_STEPS:
                filename = f"snap_step_{sim_step}.png"
                extent = ax_sim_grid.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(filename, bbox_inches=extent)
                saved_snapshots.append(filename)
                
            sim_step += 1
            
            # Move Agents
            for t in agents:
                t.x = (t.x + random.choice([-1, 0, 1])) % GRID_SIZE
                t.y = (t.y + random.choice([-1, 0, 1])) % GRID_SIZE
                
                # Check neighbors
                n_cnt = 0
                if grid_sim[(t.x+1)%GRID_SIZE, t.y] == 1: n_cnt += 1
                if grid_sim[(t.x-1)%GRID_SIZE, t.y] == 1: n_cnt += 1
                if grid_sim[t.x, (t.y+1)%GRID_SIZE] == 1: n_cnt += 1
                if grid_sim[t.x, (t.y-1)%GRID_SIZE] == 1: n_cnt += 1
                
                here = grid_sim[t.x, t.y]
                
                # Apply Genes
                if not t.carrying and here == 1:
                    if random.random() < genes_pick[n_cnt]:
                        t.carrying = True
                        grid_sim[t.x, t.y] = 0
                elif t.carrying and here == 0:
                    if random.random() < genes_drop[n_cnt]:
                        t.carrying = False
                        grid_sim[t.x, t.y] = 1
        
        # Draw updates
        display = grid_sim.copy()
        for t in agents: display[t.x, t.y] = 3 if t.carrying else 2
        img.set_data(display)
        
        # Update Metric Graph
        current_metric = calculate_order_fast(grid_sim, np.sum(grid_sim))
        sim_history_steps.append(sim_step)
        sim_history_metric.append(current_metric)
        line_sim.set_data(sim_history_steps, sim_history_metric)
        return [img, line_sim]

    # -----------------------------
    # PHASE 4: GALLERY (The Result)
    # -----------------------------
    elif current_state == STATE_GALLERY:
        show_gallery_phase()
        current_state = STATE_DONE 
        return []

    return []

if __name__ == "__main__":
    print("Launching Termite Evolution Experiment...")
    ani = animation.FuncAnimation(fig, update, interval=1, blit=False)
    plt.show()