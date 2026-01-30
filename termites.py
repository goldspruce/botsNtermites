    import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import sys

# ==========================================
# CONFIGURATION
# ==========================================
GRID_SIZE = 50          # 50x50 World
NUM_TERMITES = 100      # 100 Agents
DENSITY_BRICKS = 0.3    # 30% filled with bricks
TOTAL_STEPS = 100000    # Run 100,000 steps
STEPS_PER_FRAME = 500   # Speed up
CHECKPOINTS = [0, 25000, 50000, 75000, 100000] 

# ==========================================
# INITIALIZATION
# ==========================================
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
snapshots = []
current_step = 0

# Metric History
history_steps = []
history_metric = []

# State Flags
simulation_started = False  
simulation_finished = False 

# Randomly fill grid
random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
grid[random_matrix < DENSITY_BRICKS] = 1
total_bricks = np.sum(grid) # Count total bricks for metric calculation

# Capture initial state (Step 0)
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
def calculate_order():
    """Calculates what % of brick sides are touching other bricks."""
    if total_bricks == 0: return 0
    
    # Efficient numpy neighbor count
    # Roll grid to check neighbors (Up, Down, Left, Right)
    neighbors = np.zeros_like(grid)
    neighbors += np.roll(grid, 1, axis=0)  # Down
    neighbors += np.roll(grid, -1, axis=0) # Up
    neighbors += np.roll(grid, 1, axis=1)  # Right
    neighbors += np.roll(grid, -1, axis=1) # Left
    
    # Only count neighbors where there is actually a brick in the center
    matches = np.sum(neighbors * grid)
    
    # Max possible connections = Total Bricks * 4 sides
    max_connections = total_bricks * 4
    
    return matches / max_connections

# ==========================================
# VISUALIZATION SETUP
# ==========================================
# Create 2 subplots: Left for Grid, Right for Graph
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# --- PLOT 1: THE GRID ---
ax1.axis('off')
cmap = plt.matplotlib.colors.ListedColormap(['white', 'firebrick', 'cornflowerblue', 'lime'])
bounds = [0, 1, 2, 3, 4]
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

# Prepare initial grid WITH agents visible
initial_display_grid = grid.copy()
for t in agents:
    initial_display_grid[t.x, t.y] = 2 # Blue for waiting agents

img = ax1.imshow(initial_display_grid, interpolation='nearest', cmap=cmap, norm=norm)
title_text = ax1.set_title(f"Stigmergy Simulation\nStep: 0\n(Press ENTER to Start)", fontsize=12)

# --- PLOT 2: THE METRIC GRAPH ---
line, = ax2.plot([], [], lw=2, color='firebrick')
ax2.set_xlim(0, TOTAL_STEPS)
ax2.set_ylim(0, 1.0) # Metric goes from 0.0 to 1.0
ax2.set_title("Order Metric (Clustering)")
ax2.set_xlabel("Time (Steps)")
ax2.set_ylabel("Order (0=Chaos, 1=Solid)")
ax2.grid(True)

# ==========================================
# INTERACTION HANDLER
# ==========================================
def on_key(event):
    global simulation_started, simulation_finished
    if event.key == 'enter':
        if not simulation_started and not simulation_finished:
            simulation_started = True
            title_text.set_text(f"Stigmergy Simulation\nStep: {current_step}")
            fig.canvas.draw()
        elif simulation_finished:
             plt.close(fig) 

fig.canvas.mpl_connect('key_press_event', on_key)

# ==========================================
# UPDATE FUNCTION
# ==========================================
def update(frame):
    global current_step, simulation_finished
    
    if not simulation_started:
        return [img, line]

    # Run physics in batches
    for _ in range(STEPS_PER_FRAME):
        if current_step >= TOTAL_STEPS: break
        current_step += 1
        
        if current_step in CHECKPOINTS:
            snapshots.append(grid.copy())

        # --- PHYSICS ---
        for t in agents:
            # Move
            t.x = (t.x + random.choice([-1, 0, 1])) % GRID_SIZE
            t.y = (t.y + random.choice([-1, 0, 1])) % GRID_SIZE
            
            # Sense
            neighbors = 0
            if grid[(t.x+1)%GRID_SIZE, t.y] == 1: neighbors += 1
            if grid[(t.x-1)%GRID_SIZE, t.y] == 1: neighbors += 1
            if grid[t.x, (t.y+1)%GRID_SIZE] == 1: neighbors += 1
            if grid[t.x, (t.y-1)%GRID_SIZE] == 1: neighbors += 1
            
            density = neighbors / 4.0
            here = grid[t.x, t.y]
            
            # Act
            if not t.carrying and here == 1:
                if random.random() < (1.0 - density):
                    t.carrying = True
                    grid[t.x, t.y] = 0
            elif t.carrying and here == 0:
                if random.random() < (density + 0.01):
                    t.carrying = False
                    grid[t.x, t.y] = 1

    # Update Visuals
    title_text.set_text(f"Stigmergy Simulation\nStep: {current_step} / {TOTAL_STEPS}")
    
    # 1. Grid
    display_grid = grid.copy()
    for t in agents:
        display_grid[t.x, t.y] = 3 if t.carrying else 2
    img.set_data(display_grid)
    
    # 2. Graph
    current_metric = calculate_order()
    history_steps.append(current_step)
    history_metric.append(current_metric)
    line.set_data(history_steps, history_metric)
    
    # Check Stop
    if current_step >= TOTAL_STEPS and not simulation_finished:
        ani.event_source.stop()
        simulation_finished = True
        title_text.set_text(f"Simulation Complete.\nPress ENTER to view Timeline.")
        fig.canvas.draw()

    return [img, line]

# ==========================================
# TIMELINE
# ==========================================
def show_timeline():
    print("Generating Timeline...")
    fig2, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig2.suptitle("Evolution of Structure (Timeline)", fontsize=16)
    
    while len(snapshots) < 5:
        snapshots.append(grid.copy())

    titles = ["Start (0)", "25,000", "50,000", "75,000", "Final (100,000)"]

    for i in range(5):
        axes[i].imshow(snapshots[i], cmap='Reds', interpolation='nearest')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("-" * 40)
    print("STIGMERGY SIMULATION WITH METRICS")
    print("-" * 40)
    print("Follow instructions in the simulation window.")
    
    ani = animation.FuncAnimation(fig, update, interval=1, blit=False)
    plt.show() 

    print("Showing timeline.")
    show_timeline()