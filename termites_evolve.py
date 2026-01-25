import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import os
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
GRID_SIZE = 50          
NUM_TERMITES = 100      
DENSITY_BRICKS = 0.3    
TOTAL_STEPS = 60000     # Longer run to allow for "merging"
STEPS_PER_FRAME = 1000  # Fast forward (1000 moves per frame)
CHECKPOINTS = [0, 15000, 30000, 45000, 60000] 

# ==========================================
# OUTPUT SETUP
# ==========================================
# Create a unique timestamped folder for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
save_dir = f"simulation_{timestamp}"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created output directory: {save_dir}/")

# ==========================================
# STATE MACHINE CONSTANTS
# ==========================================
STATE_INIT_EMPTY = 0      
STATE_INIT_AGENTS = 1     
STATE_RUNNING = 2         
STATE_DONE_AGENTS = 3     
STATE_DONE_EMPTY = 4      
STATE_EXIT = 5            

current_state = STATE_INIT_EMPTY

# ==========================================
# INITIALIZATION
# ==========================================
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
snapshots = []
current_step = 0

# Randomly fill grid
random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
grid[random_matrix < DENSITY_BRICKS] = 1
total_bricks = np.sum(grid)

snapshots.append(grid.copy())

class Termite:
    def __init__(self):
        self.x = random.randint(0, GRID_SIZE-1)
        self.y = random.randint(0, GRID_SIZE-1)
        self.carrying = False

agents = [Termite() for _ in range(NUM_TERMITES)]

# ==========================================
# METRIC CALCULATION
# ==========================================
def calculate_order(target_grid=None):
    if target_grid is None: target_grid = grid
    if total_bricks == 0: return 0
    
    neighbors = np.zeros_like(target_grid)
    neighbors += np.roll(target_grid, 1, axis=0)  
    neighbors += np.roll(target_grid, -1, axis=0) 
    neighbors += np.roll(target_grid, 1, axis=1)  
    neighbors += np.roll(target_grid, -1, axis=1) 
    
    matches = np.sum(neighbors * target_grid)
    return matches / (total_bricks * 4)

history_steps = [0]
history_metric = [calculate_order(grid)]

# ==========================================
# VISUALIZATION SETUP
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

cmap = plt.matplotlib.colors.ListedColormap(['white', 'firebrick', 'cornflowerblue', 'lime'])
bounds = [0, 1, 2, 3, 4]
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

# Panel 1: Grid
img = ax1.imshow(grid, interpolation='nearest', cmap=cmap, norm=norm)
title_text = ax1.set_title(f"1. Initial State (Bricks Only)\n(Press ENTER to Add Agents)", fontsize=11)
ax1.axis('off') 

# Panel 2: Graph
line, = ax2.plot(history_steps, history_metric, lw=2, color='firebrick')
ax2.set_xlim(0, TOTAL_STEPS)
ax2.set_ylim(0.30, 0.85)     # <--- Goal range included
ax2.set_title("Order Metric (Clustering)")
ax2.set_xlabel("Time (Steps)")
ax2.grid(True)
# Add a dashed line for our goal
ax2.axhline(y=0.75, color='green', linestyle='--', alpha=0.5, label='Goal (0.75)')
ax2.legend()

# ==========================================
# LOGIC UPDATE
# ==========================================
def update(frame):
    global current_step, current_state
    
    if current_state == STATE_INIT_EMPTY:
        img.set_data(grid)
        return [img, line]

    if current_state == STATE_INIT_AGENTS:
        display = grid.copy()
        for t in agents: display[t.x, t.y] = 2 
        img.set_data(display)
        return [img, line]

    if current_state == STATE_RUNNING:
        
        save_checkpoint_name = None # Flag to save at end of frame if checkpoint hit
        
        for _ in range(STEPS_PER_FRAME):
            if current_step >= TOTAL_STEPS: break 
            current_step += 1
            if current_step in CHECKPOINTS: 
                snapshots.append(grid.copy())
                # Prepare filename for checkpoint
                save_checkpoint_name = f"screenshot_step_{current_step:05d}.png"

            for t in agents:
                t.x = (t.x + random.choice([-1, 0, 1])) % GRID_SIZE
                t.y = (t.y + random.choice([-1, 0, 1])) % GRID_SIZE
                
                # --- NEW AGGRESSIVE SENSING ---
                # Check neighbors (Up, Down, Left, Right)
                n_cnt = 0
                if grid[(t.x+1)%GRID_SIZE, t.y] == 1: n_cnt += 1
                if grid[(t.x-1)%GRID_SIZE, t.y] == 1: n_cnt += 1
                if grid[t.x, (t.y+1)%GRID_SIZE] == 1: n_cnt += 1
                if grid[t.x, (t.y-1)%GRID_SIZE] == 1: n_cnt += 1
                
                here = grid[t.x, t.y] 
                
                # --- AGGRESSIVE RULES ---
                
                # Rule 1: PICK UP
                # If I am empty-handed and standing on a brick:
                if not t.carrying and here == 1:
                    # If it has 0 or 1 neighbor (Dust/Endpoints): Definitely pick up
                    if n_cnt <= 1:
                        t.carrying = True
                        grid[t.x, t.y] = 0
                    # If it has 2 neighbors (Lines/Corners): 5% chance to break it (Destabilize small structures)
                    elif n_cnt == 2:
                        if random.random() < 0.05: 
                            t.carrying = True
                            grid[t.x, t.y] = 0
                    # If 3 or 4 neighbors (Solid): NEVER touch it.
                
                # Rule 2: DROP
                # If I am holding a brick and standing on empty ground:
                elif t.carrying and here == 0:
                    # Only drop if I find a friend (at least 1 neighbor)
                    if n_cnt >= 1:
                        # Higher chance to drop if it creates a solid block (high neighbors)
                        # n_cnt 1: 50% drop
                        # n_cnt 2: 80% drop
                        # n_cnt 3/4: 99% drop
                        chance = 0.5 + (n_cnt * 0.15)
                        if random.random() < chance:
                            t.carrying = False
                            grid[t.x, t.y] = 1 

        # Update Visuals
        title_text.set_text(f"3. Running... Step: {current_step}")
        display = grid.copy()
        for t in agents: display[t.x, t.y] = 3 if t.carrying else 2
        img.set_data(display)
        
        history_steps.append(current_step)
        history_metric.append(calculate_order()) 
        line.set_data(history_steps, history_metric)
        
        # --- AUTO SCREENSHOT AT CHECKPOINTS ---
        # We save here (after visuals update) to ensure image matches the step
        if save_checkpoint_name:
            full_path = os.path.join(save_dir, save_checkpoint_name)
            plt.savefig(full_path)
            print(f"Saved Checkpoint: {full_path}")
        
        if current_step >= TOTAL_STEPS:
            current_state = STATE_DONE_AGENTS
            title_text.set_text("4. Simulation Complete.\n(Press ENTER to Remove Agents)")
        
        return [img, line]

    if current_state == STATE_DONE_AGENTS:
        display = grid.copy()
        for t in agents: display[t.x, t.y] = 3 if t.carrying else 2
        img.set_data(display)
        return [img, line]

    if current_state == STATE_DONE_EMPTY:
        img.set_data(grid)
        return [img, line]

    return [img, line]

# ==========================================
# INTERACTION & TIMELINE
# ==========================================
def on_key(event):
    global current_state
    
    # helper to save clean screenshots before transitions
    def save_state_shot(suffix):
        filename = f"screenshot_state_{suffix}.png"
        full_path = os.path.join(save_dir, filename)
        plt.savefig(full_path)
        print(f"Saved State: {full_path}")

    if event.key == 'enter':
        if current_state == STATE_INIT_EMPTY:
            save_state_shot("01_initial") # Screenshot before adding agents
            current_state = STATE_INIT_AGENTS
            title_text.set_text(f"2. Agents Added\n(Press ENTER to Start)")
            
        elif current_state == STATE_INIT_AGENTS:
            save_state_shot("02_ready") # Screenshot before running
            current_state = STATE_RUNNING
            title_text.set_text(f"3. Running...\nStep: {current_step}")
            
        elif current_state == STATE_DONE_AGENTS:
            save_state_shot("03_finished") # Screenshot before removing agents
            current_state = STATE_DONE_EMPTY
            title_text.set_text(f"5. Agents Hidden\n(Press ENTER to View Timeline)")
            
        elif current_state == STATE_DONE_EMPTY:
            save_state_shot("04_final_grid") # Screenshot of grid before exit
            current_state = STATE_EXIT
            plt.close(fig) 

fig.canvas.mpl_connect('key_press_event', on_key)

def show_timeline():
    print("Generating Timeline...")
    fig2, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig2.suptitle("Evolution of Structure (Timeline)", fontsize=16)
    while len(snapshots) < 5: snapshots.append(grid.copy())
    
    time_labels = ["Start", "15k", "30k", "45k", "Final"]

    for i in range(5):
        snap_metric = calculate_order(snapshots[i])
        axes[i].imshow(snapshots[i], cmap='Reds', interpolation='nearest')
        axes[i].set_title(f"{time_labels[i]}\nOrder: {snap_metric:.3f}", fontsize=10)
        axes[i].axis('off') 
    plt.tight_layout() 
    
    # Save the final timeline screen automatically
    timeline_path = os.path.join(save_dir, "screenshot_05_timeline_final.png")
    plt.savefig(timeline_path)
    print(f"Saved Timeline: {timeline_path}")
    
    plt.show() 

if __name__ == "__main__":
    print("-" * 40)
    print("AGGRESSIVE STIGMERGY (Goal: > 0.75)")
    print("-" * 40)
    print(f"Images will be saved in directory: {save_dir}/")
    ani = animation.FuncAnimation(fig, update, interval=1, blit=False)
    plt.show() 
    if current_state == STATE_EXIT: show_timeline()