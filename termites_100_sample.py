import numpy as np
import random
import copy
import os
from datetime import datetime

# ==========================================
# 0. CONFIGURATION & FOLDER SETUP
# ==========================================
# We create a new folder for this batch run to keep things organized
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FOLDER = f"batch_experiment_{TIMESTAMP}"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

LOG_FILE_PATH = os.path.join(OUTPUT_FOLDER, "batch_results_log.txt")

print(f"--- INITIALIZING BATCH RUN ---")
print(f"Output file: {LOG_FILE_PATH}")

# ==========================================
# 1. EXPERIMENTAL SETTINGS
# ==========================================
# Simulation Constants
GRID_SIZE = 50          
NUM_TERMITES = 100      
DENSITY_BRICKS = 0.3    

# Evolution Settings
POPULATION_SIZE = 20    
GENERATIONS = 20        
STEPS_TRAINING = 4000   
TRIALS_PER_GENOME = 5   

# Batch Settings
TOTAL_BATCH_RUNS = 100  # WE WILL RUN THE WHOLE EXPERIMENT 100 TIMES
VALIDATION_STEPS = 10000 # The final test for the "Champion"

# ==========================================
# 2. HELPER FUNCTIONS (Math & Logic)
# ==========================================

def calculate_order_fast(grid, total_bricks):
    """
    Calculates the 'Clustering Metric' (0.0 to 1.0).
    Logic: Checks how many neighbors (Up, Down, Left, Right) contain bricks.
    """
    if total_bricks == 0: return 0
    # Shift grid to check all 4 neighbors simultaneously
    neighbors = (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                 np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))
    # Multiply by grid to only count neighbors of actual bricks
    matches = np.sum(neighbors * grid)
    # Normalize: Max possible connections is 4 * total_bricks
    return matches / (total_bricks * 4)

def run_simulation(genome, steps, return_full_stats=False):
    """
    Runs a complete simulation.
    Args:
        genome: The probability genes [Pick0-4, Drop0-4]
        steps: How long to run
        return_full_stats: If True, returns (start_metric, end_metric). 
                           If False, returns just end_metric (for evolution speed).
    """
    # 1. Setup Grid
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
    grid[random_matrix < DENSITY_BRICKS] = 1 
    total_bricks = np.sum(grid)
    
    # 2. Decode Genome
    genes_pick = genome[0:5]  
    genes_drop = genome[5:10] 
    
    # 3. Setup Agents
    agent_x = np.random.randint(0, GRID_SIZE, NUM_TERMITES)
    agent_y = np.random.randint(0, GRID_SIZE, NUM_TERMITES)
    agent_carrying = np.zeros(NUM_TERMITES, dtype=bool) 
    
    # Measure Starting Metric
    start_metric = calculate_order_fast(grid, total_bricks)
    
    # 4. Main Simulation Loop
    for _ in range(steps):
        # Move Agents
        shift_x = np.random.choice([-1, 0, 1], NUM_TERMITES)
        shift_y = np.random.choice([-1, 0, 1], NUM_TERMITES)
        agent_x = (agent_x + shift_x) % GRID_SIZE 
        agent_y = (agent_y + shift_y) % GRID_SIZE
        
        # Agent Logic (Pick/Drop)
        for i in range(NUM_TERMITES):
            x, y = agent_x[i], agent_y[i]
            
            # Count neighbors
            n_cnt = 0
            if grid[(x+1)%GRID_SIZE, y] == 1: n_cnt += 1
            if grid[(x-1)%GRID_SIZE, y] == 1: n_cnt += 1
            if grid[x, (y+1)%GRID_SIZE] == 1: n_cnt += 1
            if grid[x, (y-1)%GRID_SIZE] == 1: n_cnt += 1
            here = grid[x, y]
            
            if not agent_carrying[i] and here == 1:
                if random.random() < genes_pick[n_cnt]:
                    agent_carrying[i] = True 
                    grid[x, y] = 0           
            elif agent_carrying[i] and here == 0:
                if random.random() < genes_drop[n_cnt]:
                    agent_carrying[i] = False 
                    grid[x, y] = 1            
    
    end_metric = calculate_order_fast(grid, total_bricks)
    
    if return_full_stats:
        return start_metric, end_metric
    return end_metric

def create_genome():
    return [random.random() for _ in range(10)]

def mutate(genome):
    idx = random.randint(0, 9)          
    change = random.uniform(-0.2, 0.2)  
    genome[idx] = max(0.0, min(1.0, genome[idx] + change))
    return genome

def format_genome_str(genome):
    p = "[" + ", ".join([f"{g:.2f}" for g in genome[0:5]]) + "]"
    d = "[" + ", ".join([f"{g:.2f}" for g in genome[5:10]]) + "]"
    return p, d

# ==========================================
# 3. MAIN BATCH PROCESS
# ==========================================

# Initialize the log file header
with open(LOG_FILE_PATH, "w") as f:
    f.write(f"BATCH EXPERIMENT LOG - {TIMESTAMP}\n")
    f.write(f"Runs: {TOTAL_BATCH_RUNS} | Gens/Run: {GENERATIONS}\n")
    f.write(f"Grid: {GRID_SIZE}x{GRID_SIZE} | Termites: {NUM_TERMITES}\n")
    f.write("========================================\n\n")

# Store stats for all 100 runs here
batch_data = {
    "start_metrics": [],
    "end_metrics": [],
    "differences": []
}

print(f"Starting {TOTAL_BATCH_RUNS} runs. This may take a while...")

for run_idx in range(1, TOTAL_BATCH_RUNS + 1):
    
    # --- PHASE A: EVOLUTION (Find the best strategy) ---
    print(f"Run {run_idx}/{TOTAL_BATCH_RUNS}: Evolving...", end="\r")
    
    population = [create_genome() for _ in range(POPULATION_SIZE)]
    best_genome_of_run = None
    best_score_of_run = 0.0
    
    # We log the evolution details to file as requested
    with open(LOG_FILE_PATH, "a") as f:
        f.write(f"--- RUN #{run_idx} ---\n")

    for gen in range(1, GENERATIONS + 1):
        scores = []
        # Evaluate Population
        for genome in population:
            # We average 5 trials to ensure reliability
            trial_scores = [run_simulation(genome, STEPS_TRAINING, False) for _ in range(TRIALS_PER_GENOME)]
            avg_score = sum(trial_scores) / len(trial_scores)
            scores.append((avg_score, genome))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        best_gen_score, best_gen_genome = scores[0]
        
        # Track the all-time best for this specific run
        if best_gen_score > best_score_of_run:
            best_score_of_run = best_gen_score
            best_genome_of_run = best_gen_genome
        
        # Log Generation stats to file (User Requirement: "same output as above")
        with open(LOG_FILE_PATH, "a") as f:
            p_str, d_str = format_genome_str(best_gen_genome)
            f.write(f"Gen {gen}: Best Score {best_gen_score:.4f}\n")
            # We don't log full params every gen to save space, but only final best
            
        # Breeding (Elitism + Mutation)
        top_count = int(POPULATION_SIZE * 0.2)
        survivors = [s[1] for s in scores[:top_count]]
        next_gen = survivors[:]
        while len(next_gen) < POPULATION_SIZE:
            parent = random.choice(survivors)
            next_gen.append(mutate(copy.deepcopy(parent)))
        population = next_gen

    # --- PHASE B: FINAL VALIDATION (Test the winner) ---
    # Now we take the champion of this evolution run and test it on a fresh grid
    # for the full duration (10,000 steps).
    
    start_m, end_m = run_simulation(best_genome_of_run, VALIDATION_STEPS, True)
    diff_m = end_m - start_m
    
    # Store Data
    batch_data["start_metrics"].append(start_m)
    batch_data["end_metrics"].append(end_m)
    batch_data["differences"].append(diff_m)
    
    # Log Result of this Run
    p_final, d_final = format_genome_str(best_genome_of_run)
    with open(LOG_FILE_PATH, "a") as f:
        f.write(f"\n> RUN #{run_idx} RESULT:\n")
        f.write(f"  Genome P: {p_final}\n  Genome D: {d_final}\n")
        f.write(f"  Start Metric: {start_m:.4f}\n")
        f.write(f"  End Metric:   {end_m:.4f}\n")
        f.write(f"  Difference:   {diff_m:+.4f}\n")
        f.write("----------------------------------------\n\n")

print(f"\n\nDone! Computing Statistics...")

# ==========================================
# 4. STATISTICAL ANALYSIS
# ==========================================
# Convert lists to numpy arrays for easy math
starts = np.array(batch_data["start_metrics"])
ends = np.array(batch_data["end_metrics"])
diffs = np.array(batch_data["differences"])

# Calculate Stats
stats_output = f"""
:
##################################################
STATISTICAL ANALYSIS (N={TOTAL_BATCH_RUNS})
##################################################

1. STARTING CONDITIONS (Random Placement)
   Mean:   {np.mean(starts):.4f}
   StdDev: {np.std(starts):.4f}
   Min:    {np.min(starts):.4f}
   Max:    {np.max(starts):.4f}

2. ENDING CONDITIONS (After {VALIDATION_STEPS} steps)
   Mean:   {np.mean(ends):.4f}
   Median: {np.median(ends):.4f}
   StdDev: {np.std(ends):.4f}
   Min:    {np.min(ends):.4f}
   Max:    {np.max(ends):.4f}

3. IMPROVEMENT (End - Start)
   Mean Improvement:   {np.mean(diffs):+.4f}
   StdDev Improvement: {np.std(diffs):.4f}
   Best Run Gain:      {np.max(diffs):+.4f}
   Worst Run Gain:     {np.min(diffs):+.4f}

##################################################
"""

# Print to Console
print(stats_output)

# Append to Log File
with open(LOG_FILE_PATH, "a") as f:
    f.write(stats_output)

print(f"Full results saved to: {LOG_FILE_PATH}")