import numpy as np
import random
import time
import multiprocessing
import concurrent.futures
from scipy import stats

# ==========================================
# 1. CONFIGURATION
# ==========================================
GRID_SIZE = 50          
NUM_TERMITES = 100      
DENSITY_BRICKS = 0.3    
TOTAL_STEPS = 1000      

# ==========================================
# 2. METRIC CALCULATION
# ==========================================
def calculate_order(grid, total_bricks):
    """
    Calculates the 'Clustering Score' (0.0 to 1.0).
    We use numpy.roll to instantly shift the entire grid up, down, left, and right.
    This allows us to check the neighbors of every single cell simultaneously 
    without needing a slow, nested 'for' loop.
    """
    if total_bricks == 0: return 0
    neighbors = (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                 np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))
    matches = np.sum(neighbors * grid)
    return matches / (total_bricks * 4)

# ==========================================
# 3. CORE SIMULATION ENGINE (Self-Contained)
# ==========================================
def run_trial(is_control):
    """
    Runs a single simulation from start to finish.
    
    CRITICAL MULTIPROCESSING RULE: This function must be entirely self-contained.
    It creates its own grid, its own agents, and runs its own loop. It cannot 
    rely on global variables outside this function, because each CPU core will 
    be running its own isolated copy of this function in separate memory.
    """
    
    # 1. Setup Local Grid (Isolated to this specific CPU process)
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
    grid[random_matrix < DENSITY_BRICKS] = 1
    total_bricks = np.sum(grid)

    # 2. Setup Local Agents
    class Termite:
        def __init__(self):
            self.x = random.randint(0, GRID_SIZE-1)
            self.y = random.randint(0, GRID_SIZE-1)
            self.carrying = False

    agents = [Termite() for _ in range(NUM_TERMITES)]

    # 3. Execution Loop
    for _ in range(TOTAL_STEPS):
        for t in agents:
            # Move Randomly (Brownian motion)
            t.x = (t.x + random.choice([-1, 0, 1])) % GRID_SIZE
            t.y = (t.y + random.choice([-1, 0, 1])) % GRID_SIZE
            
            here = grid[t.x, t.y]
            
            # Read Environment (Control vs Active)
            if is_control:
                # Blind condition: The termite always thinks density is 50%
                density = 0.5 
            else:
                # Active condition: The termite counts its immediate neighbors
                n_cnt = 0
                if grid[(t.x+1)%GRID_SIZE, t.y] == 1: n_cnt += 1
                if grid[(t.x-1)%GRID_SIZE, t.y] == 1: n_cnt += 1
                if grid[t.x, (t.y+1)%GRID_SIZE] == 1: n_cnt += 1
                if grid[t.x, (t.y-1)%GRID_SIZE] == 1: n_cnt += 1
                density = n_cnt / 4.0
            
            # Decision Matrix (Probabilistic Pick-up / Drop)
            if not t.carrying and here == 1:
                if random.random() < (1.0 - density):
                    t.carrying = True
                    grid[t.x, t.y] = 0
            elif t.carrying and here == 0:
                if random.random() < (density + 0.01):
                    t.carrying = False
                    grid[t.x, t.y] = 1

    # Return only the final mathematical score to the main process
    return calculate_order(grid, total_bricks)

# ==========================================
# 4. MAIN SCRIPT & MULTIPROCESSING
# ==========================================
# CRITICAL MACOS RULE: multiprocessing requires the execution code to be hidden 
# behind this 'if __name__ == "__main__":' guard. 
if __name__ == "__main__":
    print("=== TERMITE STIGMERGY MULTICORE STATISTICAL SUITE ===")
    
    # multiprocessing.cpu_count() asks your M5 chip exactly how many logical cores it has available
    num_cores = multiprocessing.cpu_count()
    print(f"Hardware Detected: Apple M-Series with {num_cores} logical cores.")
    
    try:
        num_trials = int(input("How many trials (X) would you like to run per group? (e.g., 10000): "))
    except ValueError:
        print("Invalid input. Defaulting to 1000 trials.")
        num_trials = 1000
        
    start_time = time.time()
    
    # ---------------------------------------------------------
    # PARALLEL EXECUTION: ACTIVE TRIALS
    # ---------------------------------------------------------
    print(f"\nDistributing {num_trials} ACTIVE trials across {num_cores} cores...")
    
    # ProcessPoolExecutor creates a "pool" of background Python processes (one for each core).
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # executor.map takes our function (run_trial) and a list of arguments.
        # It automatically chunks the list and hands those chunks to idle CPU cores.
        # [False] * num_trials creates a list like: [False, False, False...] 10,000 times.
        active_scores = list(executor.map(run_trial, [False] * num_trials))
        
    # ---------------------------------------------------------
    # PARALLEL EXECUTION: CONTROL TRIALS
    # ---------------------------------------------------------
    print(f"Distributing {num_trials} CONTROL trials across {num_cores} cores...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # We do the exact same thing, but pass a list of 'True' so run_trial triggers the control logic.
        control_scores = list(executor.map(run_trial, [True] * num_trials))
    
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS! All {num_trials * 2} simulations completed in {elapsed:.2f} seconds]")
    
    # ---------------------------------------------------------
    # STATISTICAL ANALYSIS
    # ---------------------------------------------------------
    # Now that we have all the scores back from the background cores, we analyze them on the main core.
    active_mean = np.mean(active_scores)
    active_std = np.std(active_scores)
    control_mean = np.mean(control_scores)
    control_std = np.std(control_scores)
    
    # scipy.stats calculates the t-statistic and the p-value
    t_stat, p_value = stats.ttest_ind(active_scores, control_scores, equal_var=False)

    # Output Results
    print("\n==================================================")
    print("               STATISTICAL ANALYSIS               ")
    print("==================================================")
    print("NULL HYPOTHESIS (H0): Termite structures are indistinguishable")
    print("from random spatial diffusion (Active = Control).\n")
    
    print(f"ACTIVE (Stigmergic) GROUP [N={num_trials}]")
    print(f"  Mean Clustering Score: {active_mean:.4f}")
    print(f"  Standard Deviation:    {active_std:.4f}\n")
    
    print(f"CONTROL (Blind) GROUP     [N={num_trials}]")
    print(f"  Mean Clustering Score: {control_mean:.4f}")
    print(f"  Standard Deviation:    {control_std:.4f}\n")
    
    print("T-TEST RESULTS:")
    print(f"  t-statistic: {t_stat:.4f}")
    if p_value == 0.0:
        print("  p-value:     < 0.0000000001 (Effectively 0)")
    else:
        print(f"  p-value:     {p_value:.4e}")
    print("==================================================")