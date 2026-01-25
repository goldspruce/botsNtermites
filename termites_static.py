import numpy as np  # "Numerical Python": Handles the grid and math very fast
import matplotlib.pyplot as plt  # "Plotting Library": Creates the window and graphs
import matplotlib.animation as animation  # Handles the moving animation
import random  # Allows us to make random choices (coin flips)
import sys  # System commands (used to exit the program if needed)
import os   # <--- ADDED: To manage folders
from datetime import datetime # <--- ADDED: To timestamp the folder

# ==========================================
# CONFIGURATION
# Settings you can change to alter the simulation
# ==========================================
GRID_SIZE = 50          # The world is a 50x50 square grid
NUM_TERMITES = 100      # We will drop 100 agents into this world
DENSITY_BRICKS = 0.3    # 30% of the grid spots will start with a red brick
TOTAL_STEPS = 1000      # <--- CHANGED: Very Short Run (1,000 steps)
STEPS_PER_FRAME = 20    # <--- CHANGED: Slowed down so you can actually see the short run happen
CHECKPOINTS = [0, 250, 500, 750, 1000] # <--- CHANGED: Save a picture every 250 steps

# ==========================================
# OUTPUT SETUP (NEW)
# Creates a folder to save our screenshots
# ==========================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"simulation_{timestamp}"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created output directory: {save_dir}/")

# ==========================================
# STATE MACHINE CONSTANTS
# These are simple labels to track "What is the program doing right now?"
# ==========================================
STATE_INIT_EMPTY = 0      # Showing the empty field of bricks
STATE_INIT_AGENTS = 1     # Showing the agents on top of the bricks (waiting to start)
STATE_RUNNING = 2         # The simulation is actively running
STATE_DONE_AGENTS = 3     # Simulation finished, showing the final result with agents
STATE_DONE_EMPTY = 4      # Simulation finished, agents removed to show structure
STATE_EXIT = 5            # Ready to close the window

current_state = STATE_INIT_EMPTY # Start at the beginning

# ==========================================
# METRIC CALCULATION (The Math)
# Calculates how "Ordered" the world is.
# Score 0.0 = Chaos (Dust). Score 1.0 = Perfection (Solid Blocks).
# ==========================================
def calculate_order(target_grid=None):
    """
    Checks every brick to see if it is touching another brick.
    If target_grid is None, it uses the live simulation grid.
    """
    if target_grid is None: 
        target_grid = grid
    
    current_bricks = np.sum(target_grid)
    if current_bricks == 0: return 0 # Avoid dividing by zero if empty
    
    # We use 'np.roll' to shift the grid in 4 directions to check neighbors efficiently
    neighbors = np.zeros_like(target_grid)
    neighbors += np.roll(target_grid, 1, axis=0)  # Check Down
    neighbors += np.roll(target_grid, -1, axis=0) # Check Up
    neighbors += np.roll(target_grid, 1, axis=1)  # Check Right
    neighbors += np.roll(target_grid, -1, axis=1) # Check Left
    
    # Multiply neighbors by the grid to ensure we only count neighbors for existing bricks
    matches = np.sum(neighbors * target_grid)
    
    # Calculate percentage: Actual Connections / Max Possible Connections
    return matches / (current_bricks * 4)

# ==========================================
# INITIALIZATION
# Setting up the memory and starting variables
# ==========================================
# Create a grid of zeros (empty white space)
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
snapshots = [] # A list to store our saved pictures
current_step = 0

# Randomly fill the grid with bricks
random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
grid[random_matrix < DENSITY_BRICKS] = 1

# Count how many bricks we have total
total_bricks = np.sum(grid)

# Save the starting look (Step 0)
snapshots.append(grid.copy())

# Initialize Graph History with Step 0 Data
# This ensures the line starts at the very beginning (0).
history_steps = [0]
history_metric = [calculate_order(grid)]

# Define what a "Termite" is
class Termite:
    def __init__(self):
        # Pick a random X and Y coordinate to drop this agent
        self.x = random.randint(0, GRID_SIZE-1)
        self.y = random.randint(0, GRID_SIZE-1)
        # Is the agent holding a brick? Start with False (No)
        self.carrying = False

# Create a list of 100 distinct Termite agents
agents = [Termite() for _ in range(NUM_TERMITES)]

# ==========================================
# VISUALIZATION SETUP
# Creating the window and the two panels (Grid and Graph)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Define Colors: 0=White, 1=Red(Brick), 2=Blue(Agent), 3=Green(Agent carrying brick)
cmap = plt.matplotlib.colors.ListedColormap(['white', 'firebrick', 'cornflowerblue', 'lime'])
bounds = [0, 1, 2, 3, 4]
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

# --- PANEL 1: THE GRID ---
img = ax1.imshow(grid, interpolation='nearest', cmap=cmap, norm=norm)
title_text = ax1.set_title(f"1. Initial State (Bricks Only)\n(Press ENTER to Add Agents)", fontsize=11)
ax1.axis('off') 

# --- PANEL 2: THE GRAPH ---
line, = ax2.plot(history_steps, history_metric, lw=2, color='firebrick')
ax2.set_xlim(0, TOTAL_STEPS) # X-axis goes from 0 to 1,000
ax2.set_ylim(0.25, 0.60)     # <--- CHANGED: New Y-Axis Range
ax2.set_title("Order Metric (Clustering)")
ax2.set_xlabel("Time (Steps)")
ax2.grid(True) 

# ==========================================
# INTERACTION HANDLER
# This function runs every time you press a key on the keyboard
# ==========================================
def on_key(event):
    global current_state
    
    # Helper to clean up filenames
    def save_moment(name):
        filename = os.path.join(save_dir, f"{name}.png")
        plt.savefig(filename)
        print(f"Saved User Action Shot: {filename}")

    if event.key == 'enter':
        
        # Start -> Add Agents
        if current_state == STATE_INIT_EMPTY:
            save_moment("01_initial_empty") # <--- CAPTURE SCREEN
            current_state = STATE_INIT_AGENTS
            title_text.set_text(f"2. Agents Added\n(Press ENTER to Start Simulation)")
            
        # Agents visible -> Start Running
        elif current_state == STATE_INIT_AGENTS:
            save_moment("02_agents_ready") # <--- CAPTURE SCREEN
            current_state = STATE_RUNNING
            title_text.set_text(f"3. Running...\nStep: {current_step}")
            
        # Finished -> Hide Agents
        elif current_state == STATE_DONE_AGENTS:
            save_moment("03_run_finished") # <--- CAPTURE SCREEN
            current_state = STATE_DONE_EMPTY
            title_text.set_text(f"5. Agents Hidden (Structure View)\n(Press ENTER to View Timeline)")
            
        # Hidden -> Close Window
        elif current_state == STATE_DONE_EMPTY:
            save_moment("04_final_structure") # <--- CAPTURE SCREEN
            current_state = STATE_EXIT
            plt.close(fig) 

fig.canvas.mpl_connect('key_press_event', on_key)

# ==========================================
# UPDATE LOOP (The Engine)
# ==========================================
def update(frame):
    global current_step, current_state
    
    # 1. Start Screen
    if current_state == STATE_INIT_EMPTY:
        img.set_data(grid)
        return [img, line]

    # 2. Waiting with Agents
    if current_state == STATE_INIT_AGENTS:
        display = grid.copy()
        for t in agents:
            display[t.x, t.y] = 2 
        img.set_data(display)
        return [img, line]

    # 3. Running Simulation
    if current_state == STATE_RUNNING:
        
        save_checkpoint = False 
        
        for _ in range(STEPS_PER_FRAME):
            if current_step >= TOTAL_STEPS: break 
            current_step += 1
            
            if current_step in CHECKPOINTS: 
                snapshots.append(grid.copy())
                save_checkpoint = True

            # --- AGENT AI LOGIC ---
            for t in agents:
                t.x = (t.x + random.choice([-1, 0, 1])) % GRID_SIZE
                t.y = (t.y + random.choice([-1, 0, 1])) % GRID_SIZE
                
                # Check neighbors
                n_cnt = 0
                if grid[(t.x+1)%GRID_SIZE, t.y] == 1: n_cnt += 1
                if grid[(t.x-1)%GRID_SIZE, t.y] == 1: n_cnt += 1
                if grid[t.x, (t.y+1)%GRID_SIZE] == 1: n_cnt += 1
                if grid[t.x, (t.y-1)%GRID_SIZE] == 1: n_cnt += 1
                
                density = n_cnt / 4.0
                here = grid[t.x, t.y] 
                
                # Stigmergy Rules
                if not t.carrying and here == 1:
                    if random.random() < (1.0 - density):
                        t.carrying = True
                        grid[t.x, t.y] = 0 
                elif t.carrying and here == 0:
                    if random.random() < (density + 0.01):
                        t.carrying = False
                        grid[t.x, t.y] = 1 

        # Update Visuals
        title_text.set_text(f"3. Running... Step: {current_step}")
        display = grid.copy()
        for t in agents:
            display[t.x, t.y] = 3 if t.carrying else 2
        img.set_data(display)
        
        # Update Graph
        history_steps.append(current_step)
        history_metric.append(calculate_order()) 
        line.set_data(history_steps, history_metric)
        
        # --- SAVE CHECKPOINT (AUTOMATIC) ---
        if save_checkpoint:
            filename = os.path.join(save_dir, f"checkpoint_step_{current_step:04d}.png")
            plt.savefig(filename)
            print(f"Saved Checkpoint: {filename}")
        
        # Check Stop Condition
        if current_step >= TOTAL_STEPS:
            current_state = STATE_DONE_AGENTS
            title_text.set_text("4. Simulation Complete.\n(Press ENTER to Remove Agents)")
        
        return [img, line]

    # 4. Finished (Agents Visible)
    if current_state == STATE_DONE_AGENTS:
        display = grid.copy()
        for t in agents:
            display[t.x, t.y] = 3 if t.carrying else 2
        img.set_data(display)
        return [img, line]

    # 5. Finished (Agents Hidden)
    if current_state == STATE_DONE_EMPTY:
        img.set_data(grid)
        return [img, line]

    return [img, line]

# ==========================================
# TIMELINE DISPLAY
# ==========================================
def show_timeline():
    print("Generating Timeline...")
    fig2, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig2.suptitle("Evolution of Structure (Timeline)", fontsize=16)
    
    while len(snapshots) < 5: snapshots.append(grid.copy())
    
    # <--- CHANGED: Updated labels for 1k steps
    time_labels = ["Start (0)", "250", "500", "750", "Final (1k)"]

    for i in range(5):
        snap_metric = calculate_order(snapshots[i])
        
        axes[i].imshow(snapshots[i], cmap='Reds', interpolation='nearest')
        axes[i].set_title(f"{time_labels[i]}\nOrder: {snap_metric:.3f}", fontsize=10)
        axes[i].axis('off') 
    
    plt.tight_layout() 
    
    # --- SAVE TIMELINE ---
    timeline_path = os.path.join(save_dir, "05_timeline_summary.png")
    plt.savefig(timeline_path)
    print(f"Saved Timeline: {timeline_path}")

    plt.show() 

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("-" * 40)
    print("STIGMERGY SIMULATION (1k Steps)")
    print("-" * 40)
    print(f"All images will be saved to: {save_dir}/")
    print("Follow the on-screen prompts.")
    
    ani = animation.FuncAnimation(fig, update, interval=1, blit=False)
    plt.show() 

    if current_state == STATE_EXIT:
        show_timeline()