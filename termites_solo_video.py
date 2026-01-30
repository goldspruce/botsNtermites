import numpy as np               # "Numerical Python": Handles the grid and math very fast
import matplotlib.pyplot as plt  # "Plotting Library": Creates the window and graphs
import matplotlib.animation as animation  # Handles the moving animation
import random                    # Allows us to make random choices (coin flips)
import os                        # Operating System: To manage folders
from datetime import datetime    # To get the current time for file naming

# ==========================================
# 1. CONFIGURATION (The "Knobs")
# ==========================================
GRID_SIZE = 50          # The world is a 50x50 square grid
NUM_TERMITES = 100      # We will drop 100 agents into this world
DENSITY_BRICKS = 0.3    # 30% of the grid spots will start with a red brick

# --- UPDATED SETTINGS ---
TOTAL_STEPS = 1000      # <--- CHANGED: Short run (1,000 steps)
STEPS_PER_FRAME = 20    # <--- CHANGED: Slowed down (only 20 steps per video frame)
CHECKPOINTS = [0, 250, 500, 750, 1000] # <--- CHANGED: Evenly spaced for 1k run

# Video Settings
VIDEO_FILENAME = "simulation_video.mp4"
VIDEO_FPS = 15          # Frames per second (Speed of video)

# ==========================================
# 2. FOLDER SETUP
# ==========================================
# Create a unique folder based on the current time
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FOLDER = f"run_{timestamp}"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print(f"--- INITIALIZING ---")
print(f"Steps: {TOTAL_STEPS} | Snapshots: {CHECKPOINTS}")
print(f"Output folder: {OUTPUT_FOLDER}/")

# ==========================================
# 3. METRIC CALCULATION (The Math)
# ==========================================
def calculate_order(grid, total_bricks):
    """
    Calculates the 'Clustering Score' (0.0 to 1.0).
    Higher number = Better, tighter piles.
    """
    if total_bricks == 0: return 0
    
    # We look at neighbors (Up, Down, Left, Right)
    neighbors = (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                 np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))
    
    # Only count neighbors if the cell itself has a brick
    matches = np.sum(neighbors * grid)
    
    # Normalize: (Actual Connections) / (Max Possible Connections)
    return matches / (total_bricks * 4)

# ==========================================
# 4. INITIALIZATION (Setup Memory)
# ==========================================
# 1. Create the grid (0 = Empty)
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# 2. Randomly scatter bricks (1 = Brick)
random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
grid[random_matrix < DENSITY_BRICKS] = 1
total_bricks = np.sum(grid)

# 3. Create Agents
class Termite:
    def __init__(self):
        self.x = random.randint(0, GRID_SIZE-1)
        self.y = random.randint(0, GRID_SIZE-1)
        self.carrying = False

agents = [Termite() for _ in range(NUM_TERMITES)]

# 4. Prepare Storage for Snapshots
# We store tuples: (StepNumber, ImageArray, MetricScore)
saved_snapshots = {} 

# Store the "Start" state immediately
start_metric = calculate_order(grid, total_bricks)

# To see agents in the snapshot, we must paint them onto a temporary grid
start_view = grid.copy()
for t in agents: start_view[t.x, t.y] = 3 if t.carrying else 2
saved_snapshots[0] = (start_view, start_metric)

# Graph History
history_steps = [0]
history_metric = [start_metric]
current_step = 0

# ==========================================
# 5. VISUALIZATION SETUP (The Window)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Define Colors: 
# 0=White (Empty), 1=Red (Brick), 2=Blue (Agent), 3=Green (Carrying Agent)
cmap = plt.matplotlib.colors.ListedColormap(['white', 'firebrick', 'cornflowerblue', 'lime'])
bounds = [0, 1, 2, 3, 4]
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

# --- PANEL 1: THE GRID ---
# We create the image object 'img' which we will update every frame
img = ax1.imshow(start_view, interpolation='nearest', cmap=cmap, norm=norm)
ax1.set_title(f"Step: 0 | Order: {start_metric:.3f}")
# Remove ugly ticks
ax1.set_xticks([])
ax1.set_yticks([])

# --- PANEL 2: THE GRAPH ---
line, = ax2.plot(history_steps, history_metric, lw=2, color='firebrick')
ax2.set_xlim(0, TOTAL_STEPS)
ax2.set_ylim(0, 1.0) 
ax2.set_title("Clustering Metric (Order)")
ax2.set_xlabel("Time (Steps)")
ax2.grid(True)

# --- SAVE STARTING SNAPSHOT (Step 0) ---
# The animation loop starts at Step 1, so we must manually save Step 0 here.
step0_path = os.path.join(OUTPUT_FOLDER, "step_0.png")
plt.savefig(step0_path)
print(f"   [Snapshot] Saved step 0")

# ==========================================
# 6. UPDATE LOOP (The Engine)
# ==========================================
def update(frame):
    global current_step, grid, agents
    
    # Stop if we are done
    if current_step >= TOTAL_STEPS:
        return [img, line]

    # Run multiple simulation steps for every single video frame (to make it faster)
    for _ in range(STEPS_PER_FRAME):
        current_step += 1
        
        # --- AGENT LOGIC (The Brains) ---
        for t in agents:
            # 1. Move Randomly
            t.x = (t.x + random.choice([-1, 0, 1])) % GRID_SIZE
            t.y = (t.y + random.choice([-1, 0, 1])) % GRID_SIZE
            
            # 2. Check Surroundings
            n_cnt = 0
            if grid[(t.x+1)%GRID_SIZE, t.y] == 1: n_cnt += 1
            if grid[(t.x-1)%GRID_SIZE, t.y] == 1: n_cnt += 1
            if grid[t.x, (t.y+1)%GRID_SIZE] == 1: n_cnt += 1
            if grid[t.x, (t.y-1)%GRID_SIZE] == 1: n_cnt += 1
            
            density = n_cnt / 4.0
            here = grid[t.x, t.y]
            
            # 3. Decision: Pick Up or Drop? (Stigmergy)
            # If here is a brick and I'm empty: Pick up if lonely (low density)
            if not t.carrying and here == 1:
                if random.random() < (1.0 - density):
                    t.carrying = True
                    grid[t.x, t.y] = 0
            # If here is empty and I'm carrying: Drop if crowded (high density)
            elif t.carrying and here == 0:
                if random.random() < (density + 0.01):
                    t.carrying = False
                    grid[t.x, t.y] = 1
        
        # --- SNAPSHOT LOGIC ---
        if current_step in CHECKPOINTS:
            # Calculate score
            m = calculate_order(grid, total_bricks)
            
            # Create a view that includes agents (Blue/Green)
            snap_view = grid.copy()
            for t in agents: snap_view[t.x, t.y] = 3 if t.carrying else 2
            
            # Save data for later composite
            saved_snapshots[current_step] = (snap_view, m)
            
            # Save individual image to disk
            filename = os.path.join(OUTPUT_FOLDER, f"step_{current_step}.png")
            
            # Temporarily update the main display to save this exact frame
            img.set_data(snap_view)
            ax1.set_title(f"Step: {current_step} | Order: {m:.3f}")
            plt.savefig(filename)
            print(f"   [Snapshot] Saved step {current_step}")

    # --- UPDATE DISPLAY ---
    # Create the display grid (Bricks + Agents)
    display = grid.copy()
    for t in agents: display[t.x, t.y] = 3 if t.carrying else 2
    
    img.set_data(display)
    
    # Calculate current score for the graph
    curr_metric = calculate_order(grid, total_bricks)
    ax1.set_title(f"Step: {current_step} | Order: {curr_metric:.3f}")
    
    # Update Graph Line
    history_steps.append(current_step)
    history_metric.append(curr_metric)
    line.set_data(history_steps, history_metric)
    
    return [img, line]

# ==========================================
# 7. MAIN EXECUTION & SAVING
# ==========================================
# Calculate how many frames we need for the video
# (Total Steps / Steps Per Frame) + Buffer
total_frames = int(TOTAL_STEPS / STEPS_PER_FRAME) + 5

print(f"Starting Simulation ({TOTAL_STEPS} steps)...")
print(f"Rendering ~{total_frames} frames to video...")

# Create the Animation Object
ani = animation.FuncAnimation(
    fig, update, 
    frames=total_frames, 
    interval=1, blit=False, repeat=False
)

# SAVE THE VIDEO
video_path = os.path.join(OUTPUT_FOLDER, VIDEO_FILENAME)
try:
    # We use 'ffmpeg' to compress the video. 
    ani.save(video_path, writer='ffmpeg', fps=VIDEO_FPS)
    print(f"Video saved successfully: {video_path}")
except Exception as e:
    # If ffmpeg is not installed, we fallback to GIF (slower, larger file)
    print(f"FFMpeg not found ({e}). Saving as GIF instead...")
    gif_path = video_path.replace(".mp4", ".gif")
    ani.save(gif_path, writer='pillow', fps=VIDEO_FPS)
    print(f"GIF saved: {gif_path}")

# ==========================================
# 8. CREATE COMPOSITE IMAGE (The "Timeline")
# ==========================================
print("Creating Timeline Composite...")
fig_comp = plt.figure(figsize=(15, 4))
axes = [fig_comp.add_subplot(1, 5, i+1) for i in range(5)]

# Loop through our checkpoints [0, 250, 500, 750, 1000]
for i, step in enumerate(CHECKPOINTS):
    ax = axes[i]
    
    if step in saved_snapshots:
        data, metric = saved_snapshots[step]
        
        # Draw the grid
        ax.imshow(data, interpolation='nearest', cmap=cmap, norm=norm)
        ax.set_title(f"Step {step}\nOrder: {metric:.3f}", fontsize=10)
        
        # Add a black border
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
    else:
        ax.text(0.5, 0.5, "Missing", ha='center')
    
    # Turn off numbers on axis
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
timeline_path = os.path.join(OUTPUT_FOLDER, "final_timeline.png")
plt.savefig(timeline_path)
print(f"Timeline saved: {timeline_path}")

print("--- DONE ---")