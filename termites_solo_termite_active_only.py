import numpy as np
import random
import time
import multiprocessing
import concurrent.futures
from scipy import stats
import csv
import sys
from datetime import datetime

# ==========================================
# 1. CONFIGURATION (The Global Knobs)
# ==========================================
GRID_SIZE = 50          # The arena is a 50x50 pixel square
NUM_TERMITES = 100      # The number of agents moving around the grid
DENSITY_BRICKS = 0.3    # 30% of the arena starts covered in mud/bricks
TOTAL_STEPS = 1000      # How many "turns" each simulation runs before stopping

# ==========================================
# 2. METRIC CALCULATION (The Math)
# ==========================================
def calculate_order(grid, total_bricks):
    """
    Calculates the 'Clustering Score' (0.0 to 1.0).
    We use numpy.roll to instantly shift the entire grid up, down, left, and right,
    allowing us to count the neighbors for all 2,500 cells simultaneously.
    """
    if total_bricks == 0: return 0
    
    # Sum the values of the cells immediately adjacent to every cell
    neighbors = (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                 np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))
    
    # Multiply by the grid so we ONLY count neighbors if the cell itself has a brick
    matches = np.sum(neighbors * grid)
    
    # Divide by the maximum possible connections to get a neat 0 to 1 score
    return matches / (total_bricks * 4)

# ==========================================
# 3. CORE SIMULATION ENGINE (Active Only)
# ==========================================
def run_trial():
    """
    Runs a single 1,000-step simulation of ACTIVE termites.
    Takes no arguments, as we no longer have a blind/control group.
    """
    # 1. Generate the initial random grid
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
    grid[random_matrix < DENSITY_BRICKS] = 1
    total_bricks = np.sum(grid)

    # ---> TRACKING INITIAL STATE <---
    # Capture the starting disorder of this specific grid before the termites touch it.
    initial_score = calculate_order(grid, total_bricks)

    # 2. Spawn the agents
    class Termite:
        def __init__(self):
            self.x = random.randint(0, GRID_SIZE-1)
            self.y = random.randint(0, GRID_SIZE-1)
            self.carrying = False

    agents = [Termite() for _ in range(NUM_TERMITES)]

    # 3. The main loop of time
    for _ in range(TOTAL_STEPS):
        for t in agents:
            # MOVEMENT: Random walk with wrap-around edges
            t.x = (t.x + random.choice([-1, 0, 1])) % GRID_SIZE
            t.y = (t.y + random.choice([-1, 0, 1])) % GRID_SIZE
            
            here = grid[t.x, t.y]
            
            # SENSING: Active Stigmergy (Counting immediate neighbors)
            n_cnt = 0
            if grid[(t.x+1)%GRID_SIZE, t.y] == 1: n_cnt += 1
            if grid[(t.x-1)%GRID_SIZE, t.y] == 1: n_cnt += 1
            if grid[t.x, (t.y+1)%GRID_SIZE] == 1: n_cnt += 1
            if grid[t.x, (t.y-1)%GRID_SIZE] == 1: n_cnt += 1
            density = n_cnt / 4.0 # Convert 0-4 into a 0.0-1.0 probability
            
            # ACTION: Probabilistic pick-up and drop rules
            if not t.carrying and here == 1:
                # Pick up isolated blocks
                if random.random() < (1.0 - density):
                    t.carrying = True
                    grid[t.x, t.y] = 0
            elif t.carrying and here == 0:
                # Drop near clustered blocks
                if random.random() < (density + 0.01):
                    t.carrying = False
                    grid[t.x, t.y] = 1

    # ---> TRACKING FINAL STATE <---
    # Calculate how much the score changed from step 0 to step 1000
    final_score = calculate_order(grid, total_bricks)
    delta_score = final_score - initial_score
    
    # Return the full lifecycle of the grid's order
    return initial_score, final_score, delta_score

# ==========================================
# 4. TERMINAL HELPER FUNCTIONS
# ==========================================
def print_progress_bar(iteration, total, prefix='', suffix='', length=40, fill='█'):
    """Draws a smooth, self-overwriting progress bar in the terminal."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: 
        print() 

# ==========================================
# 5. MAIN EXECUTION & MULTIPROCESSING
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*55)
    print(" TERMITE STIGMERGY: NET CHANGE STATISTICAL SUITE ")
    print("="*55)
    
    num_cores = multiprocessing.cpu_count()
    print(f"Hardware Detected: Apple M-Series with {num_cores} logical cores.\n")
    
    # USER INPUT
    try:
        user_input = input("Enter the number of trials (X) to run [default: 1000]: ")
        num_trials = int(user_input) if user_input.strip() != "" else 1000
        if num_trials <= 0: raise ValueError
    except ValueError:
        print("Invalid input. Defaulting to 1000 trials.")
        num_trials = 1000

    print(f"\nInitializing {num_trials} active simulations...")
    print(f"Offloading to {num_cores} background CPU cores...\n")
    
    # DATA STRUCTURES
    initial_scores = []
    final_scores = []
    delta_scores = []
    raw_results = [] 
    
    start_time = time.time()
    completed_tasks = 0
    
    print_progress_bar(0, num_trials, prefix='Progress:', suffix='Complete', length=40)
    
    # SPIN UP THE M5 MULTIPROCESSING POOL
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        
        # Submit the tasks (No arguments needed for run_trial anymore)
        futures = [executor.submit(run_trial) for _ in range(num_trials)]
        
        # COLLECT RESULTS AS THEY FINISH
        for future in concurrent.futures.as_completed(futures):
            init_val, final_val, delta_val = future.result()
            
            initial_scores.append(init_val)
            final_scores.append(final_val)
            delta_scores.append(delta_val)
            
            raw_results.append([init_val, final_val, delta_val])
            
            completed_tasks += 1
            print_progress_bar(completed_tasks, num_trials, prefix='Progress:', suffix='Complete', length=40)

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS! All {num_trials} simulations completed in {elapsed:.2f} seconds]\n")
    
    # ==========================================
    # 6. FILE SAVING & STATISTICAL ANALYSIS
    # ==========================================
    print("Calculating statistics and saving files...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"termite_net_change_data_{timestamp}.csv"
    txt_filename = f"termite_net_change_stats_{timestamp}.txt"
    
    # Save the Raw CSV data
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Initial_Score", "Final_Score", "Delta_Score"])
        writer.writerows(raw_results)
        
    # --- THE NEW STATISTICAL TEST ---
    # Because we want to know if there was a "net change", we run a One-Sample T-Test 
    # on the Delta scores against a hypothesized population mean of 0.0 (no change).
    mean_initial = np.mean(initial_scores)
    mean_final = np.mean(final_scores)
    mean_delta = np.mean(delta_scores)
    std_delta = np.std(delta_scores)
    
    # scipy.stats.ttest_1samp tests if the mean of the deltas is significantly different from 0
    t_stat, p_value = stats.ttest_1samp(delta_scores, 0.0)
    
    p_str = "< 0.0000000001 (Effectively 0)" if p_value == 0.0 else f"{p_value:.4e}"

    # Format the final report
    report = "="*55 + "\n"
    report += "               STATISTICAL ANALYSIS               \n"
    report += "="*55 + "\n"
    report += "NULL HYPOTHESIS (H0): There is no net change in spatial order.\n"
    report += "The mean of the Delta scores (Final - Initial) equals zero.\n\n"
    
    report += f"ACTIVE (Stigmergic) GROUP [N={num_trials}]\n"
    report += f"  Mean Initial Score: {mean_initial:.4f}\n"
    report += f"  Mean Final Score:   {mean_final:.4f}\n"
    report += f"  Mean Net Change (Δ): {mean_delta:.4f}\n"
    report += f"  Std Dev of Change:   {std_delta:.4f}\n\n"
    
    report += "ONE-SAMPLE T-TEST RESULTS (Testing Δ against 0):\n"
    report += f"  t-statistic: {t_stat:.4f}\n"
    report += f"  p-value:     {p_str}\n\n"
    
    if p_value < 0.05 and mean_delta > 0:
        report += "CONCLUSION: We REJECT the Null Hypothesis.\n"
        report += "The active termites produce a statistically significant\n"
        report += "positive net increase in the order of the environment.\n"
    elif p_value < 0.05 and mean_delta < 0:
        report += "CONCLUSION: We REJECT the Null Hypothesis.\n"
        report += "Interestingly, the termites significantly DECREASED order.\n"
    else:
        report += "CONCLUSION: We FAIL TO REJECT the Null Hypothesis.\n"
        report += "Any change in order is statistically indistinguishable from zero.\n"
        
    report += "-"*55 + "\n"
    report += f"Data saved to:\n  - {csv_filename}\n  - {txt_filename}\n"
    report += "="*55 + "\n"
    
    # Save and Print
    with open(txt_filename, 'w') as txtfile:
        txtfile.write(report)
        
    print(report)