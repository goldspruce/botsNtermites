Here is the updated, unified script. It now includes the **Burn-In Phase** to allow the system to reach a thermodynamic steady state before taking the initial baseline measurement. 

To keep the code clean and strictly adhere to the DRY (Don't Repeat Yourself) principle, I packaged the termites' movement and interaction logic into a nested helper function called `step_agents()`. This allows us to easily call the exact same physical rules during both the short "burn-in" loop and the long "measurement" loop without copy-pasting the core engine.

### The Unified Multicore Suite with Burn-In (`termites_act_v_blind_burn_in.py`)

```python
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
# These parameters dictate the rules of the termite world.
GRID_SIZE = 50          # The arena is a 50x50 pixel square
NUM_TERMITES = 100      # The number of agents moving around the grid
DENSITY_BRICKS = 0.3    # 30% of the arena starts covered in mud/bricks
BURN_IN_STEPS = 100     # New: Steps to reach thermodynamic equilibrium before measuring
TOTAL_STEPS = 1000      # How many "turns" the simulation measures before stopping

# ==========================================
# 2. METRIC CALCULATION (The Math)
# ==========================================
def calculate_order(grid, total_bricks):
    """
    Calculates the 'Clustering Score' (0.0 to 1.0).
    Instead of checking each cell one-by-one with a slow 'for' loop, 
    we use numpy.roll to shift the entire grid up, down, left, and right.
    This lets us instantly count the neighbors for all 2,500 cells at once.
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
# 3. CORE SIMULATION ENGINE (The Background Task)
# ==========================================
def run_trial(is_control):
    """
    Runs a single simulation. 
    This function is completely self-contained. It builds its own grid 
    and its own agents so that each background CPU core can run it in isolated memory.
    """
    # Create a fresh 50x50 grid filled with zeros
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    
    # Randomly scatter bricks until 30% density is reached
    random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
    grid[random_matrix < DENSITY_BRICKS] = 1
    total_bricks = np.sum(grid)

    # Define our simple, mindless agent
    class Termite:
        def __init__(self):
            # Spawn at a random location, carrying nothing
            self.x = random.randint(0, GRID_SIZE-1)
            self.y = random.randint(0, GRID_SIZE-1)
            self.carrying = False

    agents = [Termite() for _ in range(NUM_TERMITES)]

    # --- HELPER FUNCTION FOR SIMULATION LOOP ---
    def step_agents():
        """Advances the simulation by one step for all agents."""
        for t in agents:
            # 1. MOVEMENT: Random walk (Brownian motion)
            # The '%' (modulo) ensures they wrap around the edges like Pac-Man
            t.x = (t.x + random.choice([-1, 0, 1])) % GRID_SIZE
            t.y = (t.y + random.choice([-1, 0, 1])) % GRID_SIZE
            
            here = grid[t.x, t.y]
            
            # 2. SENSING: Read the environment
            if is_control:
                # CONTROL: The termite is blinded. It always assumes 50% density.
                density = 0.5 
            else:
                # ACTIVE: The termite checks its immediate up/down/left/right neighbors
                n_cnt = 0
                if grid[(t.x+1)%GRID_SIZE, t.y] == 1: n_cnt += 1
                if grid[(t.x-1)%GRID_SIZE, t.y] == 1: n_cnt += 1
                if grid[t.x, (t.y+1)%GRID_SIZE] == 1: n_cnt += 1
                if grid[t.x, (t.y-1)%GRID_SIZE] == 1: n_cnt += 1
                density = n_cnt / 4.0 # Convert 0-4 into a 0.0-1.0 probability
            
            # 3. ACTION: Probabilistic pick-up and drop rules
            if not t.carrying and here == 1:
                # If block is alone (density ~0), high chance to pick it up
                if random.random() < (1.0 - density):
                    t.carrying = True
                    grid[t.x, t.y] = 0
            elif t.carrying and here == 0:
                # If block is crowded (density ~1), high chance to drop it here
                if random.random() < (density + 0.01):
                    t.carrying = False
                    grid[t.x, t.y] = 1

    # ---> NEW: THE BURN-IN PHASE <---
    # We run the simulation for a short time to let the termites pick up bricks
    # and reach a thermodynamic steady state (vapor pressure equilibrium).
    # This prevents the initial mass loss from artificially dropping the score.
    for _ in range(BURN_IN_STEPS):
        step_agents()

    # ---> DELTA LOGIC: INITIAL MEASUREMENT <---
    # Capture the starting disorder of this specific grid AFTER the burn-in phase.
    initial_score = calculate_order(grid, total_bricks)

    # ---> THE MAIN MEASUREMENT PHASE <---
    # Now we run the actual recorded experiment for the main duration.
    for _ in range(TOTAL_STEPS):
        step_agents()

    # ---> DELTA LOGIC: FINAL MEASUREMENT <---
    # Calculate how much the score changed from the post-burn-in start to the end.
    final_score = calculate_order(grid, total_bricks)
    delta_score = final_score - initial_score
    
    # Return the control flag, the final state, AND the amount of work done (delta)
    return is_control, final_score, delta_score

# ==========================================
# 4. TERMINAL HELPER FUNCTIONS
# ==========================================
def print_progress_bar(iteration, total, prefix='', suffix='', length=40, fill='█'):
    """
    Creates a dynamic progress bar in the terminal that overwrites itself 
    using the carriage return (\r) character, keeping the console clean.
    """
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    # \r brings the cursor back to the start of the line, overwriting the previous frame
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: 
        print() # Move to the next line when finished

# ==========================================
# 5. MAIN EXECUTION & MULTIPROCESSING
# ==========================================
if __name__ == "__main__":
    # Ensure macOS plays nicely by keeping execution inside this __main__ guard
    print("\n" + "="*50)
    print("   TERMITE STIGMERGY MULTICORE STATISTICAL SUITE   ")
    print("="*50)
    
    # Detect Hardware
    num_cores = multiprocessing.cpu_count()
    print(f"Hardware Detected: Apple M-Series with {num_cores} logical cores.\n")
    
    # 1. USER INPUT (Terminal I/O)
    try:
        user_input = input("Enter the number of trials (X) per group [default: 1000]: ")
        num_trials = int(user_input) if user_input.strip() != "" else 1000
        if num_trials <= 0: raise ValueError
    except ValueError:
        print("Invalid input. Defaulting to 1000 trials.")
        num_trials = 1000

    print(f"\nInitializing {num_trials * 2} total simulations...")
    print(f"Offloading tasks to {num_cores} background CPU cores...\n")
    
    # 2. SETUP DATA STRUCTURES
    active_deltas = []
    control_deltas = []
    raw_results = [] # Stores rows for the CSV
    
    start_time = time.time()
    total_tasks = num_trials * 2
    completed_tasks = 0
    
    # Initialize the progress bar at 0%
    print_progress_bar(0, total_tasks, prefix='Progress:', suffix='Complete', length=40)
    
    # 3. SPIN UP THE M5 MULTIPROCESSING POOL
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        
        # Submit all ACTIVE tasks to the pool
        active_futures = [executor.submit(run_trial, False) for _ in range(num_trials)]
        # Submit all CONTROL tasks to the pool
        control_futures = [executor.submit(run_trial, True) for _ in range(num_trials)]
        
        all_futures = active_futures + control_futures
        
        # 4. COLLECT RESULTS AS THEY FINISH
        for future in concurrent.futures.as_completed(all_futures):
            # Unpack the three returned values from our simulation
            is_control, final_score, delta_score = future.result()
            
            if is_control:
                control_deltas.append(delta_score)
                group_name = "Control (Blind)"
            else:
                active_deltas.append(delta_score)
                group_name = "Active (Stigmergic)"
            
            # Save the Delta to the CSV file
            raw_results.append([group_name, final_score, delta_score])
            
            # Increment progress and update the visual bar in the terminal
            completed_tasks += 1
            print_progress_bar(completed_tasks, total_tasks, prefix='Progress:', suffix='Complete', length=40)

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS! All {total_tasks} simulations completed in {elapsed:.2f} seconds]\n")
    
    # ==========================================
    # 6. FILE SAVING & STATISTICAL ANALYSIS
    # ==========================================
    print("Calculating statistics and saving files...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"termite_raw_data_{timestamp}.csv"
    txt_filename = f"termite_stats_{timestamp}.txt"
    
    # Save the Raw CSV data
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Group", "Final_Clustering_Score", "Delta_Score"])
        writer.writerows(raw_results)
        
    # Compute Statistics (Mean, Standard Deviation, and Independent t-test)
    active_mean, active_std = np.mean(active_deltas), np.std(active_deltas)
    control_mean, control_std = np.mean(control_deltas), np.std(control_deltas)
    t_stat, p_value = stats.ttest_ind(active_deltas, control_deltas, equal_var=False)
    
    # Handle the extremely small p-values gracefully
    p_str = "< 0.0000000001 (Effectively 0)" if p_value == 0.0 else f"{p_value:.4e}"

    # Format the final report
    report = "="*50 + "\n"
    report += "               STATISTICAL ANALYSIS               \n"
    report += "="*50 + "\n"
    report += "NULL HYPOTHESIS (H0): Termite structures are indistinguishable\n"
    report += "from random spatial diffusion (Active = Control).\n\n"
    report += "NOTE: Analysis is performed on the DELTA (Final Score - Initial Score)\n"
    report += f"to eliminate baseline noise. A {BURN_IN_STEPS}-step burn-in was used\n"
    report += "to reach thermodynamic equilibrium before taking initial measurements.\n\n"
    
    report += f"ACTIVE (Stigmergic) GROUP [N={num_trials}]\n"
    report += f"  Mean Delta Score: {active_mean:.4f}\n"
    report += f"  Std Dev:          {active_std:.4f}\n\n"
    
    report += f"CONTROL (Blind) GROUP     [N={num_trials}]\n"
    report += f"  Mean Delta Score: {control_mean:.4f}\n"
    report += f"  Std Dev:          {control_std:.4f}\n\n"
    
    report += "T-TEST RESULTS:\n"
    report += f"  t-statistic: {t_stat:.4f}\n"
    report += f"  p-value:     {p_str}\n\n"
    report += "-"*50 + "\n"
    report += f"Data saved to:\n  - {csv_filename}\n  - {txt_filename}\n"
    report += "="*50 + "\n"
    
    # Save the formatted text to the TXT file
    with open(txt_filename, 'w') as txtfile:
        txtfile.write(report)
        
    # Print the final report to the terminal
    print(report)
```