import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import copy

# ==========================================
# CONFIGURATION
# ==========================================
GRID_SIZE = 50
NUM_TERMITES = 100
DENSITY_BRICKS = 0.3
TOTAL_STEPS_VISUAL = 50000  # Long run for the final show
STEPS_PER_FRAME = 1000      # Fast forward for visualization

# GENETIC ALGORITHM SETTINGS
POPULATION_SIZE = 20
GENERATIONS = 15
STEPS_TRAINING = 4000       # Short run for evolution (speed over precision)

# ==========================================
# THE EVOLUTIONARY ENGINE (HEADLESS SIMULATION)
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
        # Move all agents (Vectorized for speed would be better, but loop is clearer for logic)
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
# GENETIC ALGORITHM FUNCTIONS
# ==========================================
def create_genome():
    # Return 10 random probabilities (0.0 to 1.0)
    return [random.random() for _ in range(10)]

def mutate(genome):
    # Mutate one gene slightly
    idx = random.randint(0, 9)
    genome[idx] = max(0.0, min(1.0, genome[idx] + random.uniform(-0.2, 0.2)))
    return genome

def run_evolution():
    print(f"🧬 STARTING DIRECTED EVOLUTION (Thesis Experiment)")
    print(f"Target: Evolve 'Type I' parameters that mimic 'Type IV' Intentionality.")
    print("-" * 60)
    
    population = [create_genome() for _ in range(POPULATION_SIZE)]
    
    for gen in range(GENERATIONS):
        scores = []
        for i, genome in enumerate(population):
            score = run_simulation_headless(genome)
            scores.append((score, genome))
        
        # Sort by fitness (Highest Order Metric is best)
        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_genome = scores[0]
        
        print(f"Gen {gen+1}/{GENERATIONS} | Best Fitness: {best_score:.4f} | Genes: {[round(x,2) for x in best_genome]}")
        
        # Selection: Keep top 20% (Elitism)
        top_count = int(POPULATION_SIZE * 0.2)
        survivors = [s[1] for s in scores[:top_count]]
        
        # Reproduction: Fill rest of population with mutated clones of survivors
        next_gen = survivors[:]
        while len(next_gen) < POPULATION_SIZE:
            parent = random.choice(survivors)
            child = mutate(copy.deepcopy(parent))
            next_gen.append(child)
            
        population = next_gen
        
    print("-" * 60)
    print("🏆 EVOLUTION COMPLETE. SIMULATING CHAMPION...")
    return scores[0][1]

# ==========================================
# VISUAL SIMULATION (THE RESULT)
# ==========================================
def run_visual_simulation(best_genome):
    # Extract Genes
    genes_pick = best_genome[0:5]
    genes_drop = best_genome[5:10]
    
    # Setup Visuals
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
    grid[random_matrix < DENSITY_BRICKS] = 1
    total_bricks = np.sum(grid)
    
    snapshots = []
    current_step = 0
    history_metric = []
    history_steps = []
    
    # Setup Agents
    class Termite:
        def __init__(self):
            self.x = random.randint(0, GRID_SIZE-1)
            self.y = random.randint(0, GRID_SIZE-1)
            self.carrying = False
    agents = [Termite() for _ in range(NUM_TERMITES)]
    
    # Plot Setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    cmap = plt.matplotlib.colors.ListedColormap(['white', 'firebrick', 'cornflowerblue', 'lime'])
    bounds = [0, 1, 2, 3, 4]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    img = ax1.imshow(grid, interpolation='nearest', cmap=cmap, norm=norm)
    title_text = ax1.set_title("Evolved Agent Simulation", fontsize=11)
    ax1.axis('off')
    
    line, = ax2.plot([], [], lw=2, color='firebrick')
    ax2.set_xlim(0, TOTAL_STEPS_VISUAL)
    ax2.set_ylim(0.3, 0.85)
    ax2.set_title("Order Metric (Best Evolved Genome)")
    ax2.grid(True)
    
    # Update Loop
    def update(frame):
        nonlocal current_step
        for _ in range(STEPS_PER_FRAME):
            if current_step >= TOTAL_STEPS_VISUAL: break
            current_step += 1
            
            for t in agents:
                t.x = (t.x + random.choice([-1, 0, 1])) % GRID_SIZE
                t.y = (t.y + random.choice([-1, 0, 1])) % GRID_SIZE
                
                x, y = t.x, t.y
                here = grid[x, y]
                
                n_cnt = 0
                if grid[(x+1)%GRID_SIZE, y] == 1: n_cnt += 1
                if grid[(x-1)%GRID_SIZE, y] == 1: n_cnt += 1
                if grid[x, (y+1)%GRID_SIZE] == 1: n_cnt += 1
                if grid[x, (y-1)%GRID_SIZE] == 1: n_cnt += 1
                
                # --- APPLY EVOLVED GENES ---
                if not t.carrying and here == 1:
                    # Gene Look-up for Pick Up
                    if random.random() < genes_pick[n_cnt]:
                        t.carrying = True
                        grid[x, y] = 0
                elif t.carrying and here == 0:
                    # Gene Look-up for Drop
                    if random.random() < genes_drop[n_cnt]:
                        t.carrying = False
                        grid[x, y] = 1

        # Update Graphics
        title_text.set_text(f"Running Evolved Species... Step: {current_step}")
        display = grid.copy()
        for t in agents: display[t.x, t.y] = 3 if t.carrying else 2
        img.set_data(display)
        
        history_steps.append(current_step)
        history_metric.append(calculate_order_fast(grid, total_bricks))
        line.set_data(history_steps, history_metric)
        
        return [img, line]

    ani = animation.FuncAnimation(fig, update, interval=1, blit=False)
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Run the Evolution (The "Experiment")
    best_genome = run_evolution()
    
    # 2. Print the Analysis (The "Thesis Insight")
    print("\n🔬 GENOME ANALYSIS (Mechanistic Interpretability):")
    print(f"PICK UP Probs (0-4 neighbors): {[round(x,2) for x in best_genome[0:5]]}")
    print(f"DROP Probs    (0-4 neighbors): {[round(x,2) for x in best_genome[5:10]]}")
    print("\nINTERPRETATION:")
    print("Note how the algorithm likely evolved high pick-up rates for 0-1 neighbors (cleaning dust)")
    print("and high drop rates for 2-3 neighbors (building walls).")
    print("The 'Intentionality' is encoded in these 10 probability floats.")
    print("\nLaunching Visual Simulation...")
    
    # 3. Run the Show
    run_visual_simulation(best_genome)