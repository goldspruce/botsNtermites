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
DENSITY_BRICKS = 0.3    # 30% filled with bricks (User Specified)
TOTAL_STEPS = 100000    # Run 100,000 steps
STEPS_PER_FRAME = 500   # Speed up: Calculate 500 moves before updating screen
CHECKPOINTS = [0, 25000, 50000, 75000, 100000] # When to save snapshots

# ==========================================
# INITIALIZATION
# ==========================================
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
snapshots = []
current_step = 0
simulation_started = False  # Flag to pause at start

# Randomly fill grid
random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
grid[random_matrix < DENSITY_BRICKS] = 1

# Capture initial state (Step 0)
snapshots.append(grid.copy())

class Termite:
    def __init__(self):
        self.x = random.randint(0, GRID_SIZE-1)
        self.y = random.randint(0, GRID_SIZE-1)
        self.carrying = False

agents = [Termite() for _ in range(NUM_TERMITES)]

# ==========================================
# VISUALIZATION SETUP
# ==========================================
fig, ax = plt.subplots(figsize=(6, 6))
ax.axis('off')

# Colors: 0=White, 1=Red, 2=Blue (Agent), 3=Green (Carrying)
cmap = plt.matplotlib.colors.ListedColormap(['white', 'firebrick', 'cornflowerblue', 'lime'])
bounds = [0, 1, 2, 3, 4]
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

img = ax.imshow(grid, interpolation='nearest', cmap=cmap, norm=norm)
title_text = ax.set_title(f"Stigmergy Simulation\nStep: 0 / {TOTAL_STEPS}\n(Press ENTER to Start)", fontsize=12)

# ==========================================
# INTERACTION HANDLER (Start on Enter)
# ==========================================
def on_key(event):
    global simulation_started
    if event.key == 'enter':
        simulation_started = True
        # Remove the instructions from title
        title_text.set_text(f"Stigmergy Simulation\nStep: {current_step} / {TOTAL_STEPS}")
        fig.canvas.draw()

# Connect the keyboard listener
fig.canvas.mpl_connect('key_press_event', on_key)

# ==========================================
# UPDATE FUNCTION (Animation Loop)
# ==========================================
def update(frame):
    global current_step
    
    # PAUSE LOGIC: Do nothing until user hits ENTER
    if not simulation_started:
        return [img]

    # Run multiple logic steps for every 1 screen refresh
    for _ in range(STEPS_PER_FRAME):
        if current_step >= TOTAL_STEPS:
            break
            
        current_step += 1
        
        # Check if we hit a checkpoint
        if current_step in CHECKPOINTS and current_step != 0:
            snapshots.append(grid.copy())

        # --- SIMULATION LOGIC (YOUR EXACT MECHANICS) ---
        for t in agents:
            # Move (Random Walk)
            t.x = (t.x + random.choice([-1, 0, 1])) % GRID_SIZE
            t.y = (t.y + random.choice([-1, 0, 1])) % GRID_SIZE
            
            # Sense (Optimized 4-neighbor check)
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

    # Update Title
    title_text.set_text(f"Stigmergy Simulation\nStep: {current_step} / {TOTAL_STEPS}")
    
    # Prepare grid for display
    display_grid = grid.copy()
    for t in agents:
        display_grid[t.x, t.y] = 3 if t.carrying else 2
        
    img.set_data(display_grid)
    
    # STOP CONDITION
    if current_step >= TOTAL_STEPS:
        ani.event_source.stop()
        plt.close(fig) # Close window automatically

    return [img]

# ==========================================
# SNAPSHOT DISPLAY FUNCTION
# ==========================================
def show_timeline():
    print("Generating Timeline...")
    fig2, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig2.suptitle("Evolution of Structure (Timeline)", fontsize=16)
    
    # Fill missing snapshots if simulation ended early/weirdly
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
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Launching Simulation Window...")
    print("NOTE: Click the window and press ENTER to start the agents.")
    
    ani = animation.FuncAnimation(fig, update, interval=1, blit=False)
    plt.show() # Code pauses here until window closes

    print("-" * 40)
    print("Simulation Complete.")
    input(">> Press ENTER in this terminal to view the Timeline Snapshots... ")
    
    show_timeline()