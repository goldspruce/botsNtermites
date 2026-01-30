import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import copy
import os
from datetime import datetime

# ==========================================
# 0. CONFIGURATION & FOLDER SETUP
# ==========================================
# We use the current time to create a unique folder for this run
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FOLDER = f"experiment_{TIMESTAMP}"

# Create the folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print(f"--- INITIALIZING ---")
print(f"All outputs will be saved to: {OUTPUT_FOLDER}/")

# ==========================================
# 1. EXPERIMENTAL SETTINGS (The "Knobs")
# ==========================================
GRID_SIZE = 50          # The arena is a 50x50 square
NUM_TERMITES = 100      # Total number of agents
DENSITY_BRICKS = 0.3    # 30% of the grid starts with wood chips (bricks)

# Evolution Settings (Genetic Algorithm)
POPULATION_SIZE = 20    # How many different "strategies" we test per generation
GENERATIONS = 20        # How many times we evolve/improve
STEPS_TRAINING = 4000   # How long we let them run during the "learning" phase
TRIALS_PER_GENOME = 5   # We test each strategy 5 times to get a fair average

# Visualization Settings
TOTAL_STEPS_VISUAL = 10000 # The final "Champion" run lasts this long
STEPS_PER_FRAME = 200      # To make video faster, we calculate 200 steps for every 1 video frame

# VIDEO SETTINGS
SAVE_VIDEO = True         
VIDEO_FILENAME = "simulation_video.mp4"
VIDEO_FPS = 30             # Frames Per Second (Standard video speed)

# Timing Controls (How long things stay on screen)
FRAMES_PER_GEN = 15        # Show each Evolution Generation for 0.5 seconds
FRAMES_PAUSE_START = 30    # Pause at Step 0 for 1 second
FRAMES_FINAL_COMPOSITE = 60 # Show the final summary for 2 seconds

# Calculate total video frames needed
# (Generations) + (Pause) + (Simulation) + (Final Summary) + (Buffer)
TOTAL_VIDEO_FRAMES = (
    (GENERATIONS * FRAMES_PER_GEN) + 
    FRAMES_PAUSE_START + 
    int(TOTAL_STEPS_VISUAL / STEPS_PER_FRAME) + 
    FRAMES_FINAL_COMPOSITE + 
    20
)

# ==========================================
# 2. GLOBAL VARIABLES (The "State" of the program)
# ==========================================
# We define these "States" so the update loop knows what to draw
STATE_EVOLVING = 0      # Phase 1: searching for best genes
STATE_TRANSITION = 1    # Setup phase between evolution and simulation
STATE_PAUSE_START = 2   # Waiting at Step 0
STATE_SIMULATING = 3    # Phase 2: Watching the champion run
STATE_COMPOSITE = 4     # Phase 3: Showing all snapshots
STATE_DONE = 5          # Finished

current_state = STATE_EVOLVING
current_gen = 1         
sim_step = 0
pause_counter = 0

# Variables to store the best results
best_genome_overall = None      
best_score_overall = 0.0        
best_info_overall = "N/A"         

# Data storage for saving images later
# Dictionary format: { step_number : (image_data, score_text, full_title) }
saved_snapshot_data = {} 
evolution_history = []
scores_gen_1 = []

# Placeholder variables for Graphics (will be filled in Section 4)
fig = None
ax_evol_text = None
ax_evol_graph = None
ax_sim_grid = None
ax_sim_graph = None
text_display = None
line_fitness = None
img = None
line_sim = None

# Create the Log File
LOG_FILE_PATH = os.path.join(OUTPUT_FOLDER, "evolution_log.txt")
with open(LOG_FILE_PATH, "w") as f:
    f.write(f"EXPERIMENT LOG - {TIMESTAMP}\n")
    f.write(f"Grid: {GRID_SIZE}x{GRID_SIZE} | Termites: {NUM_TERMITES}\n")
    f.write("========================================\n\n")

# ==========================================
# 3. HELPER FUNCTIONS (The Logic)
# ==========================================

def calculate_order_fast(grid, total_bricks):
    """
    Calculates how "organized" the grid is.
    Logic: We count how many bricks are touching other bricks.
    Higher score (closer to 1.0) means big, solid clusters.
    """
    if total_bricks == 0: return 0
    # 'roll' shifts the grid to check Up, Down, Left, Right neighbors
    neighbors = (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                 np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))
    matches = np.sum(neighbors * grid)
    # Normalize score between 0 and 1
    return matches / (total_bricks * 4)

def run_simulation_headless(genome):
    """
    Runs a FAST simulation without graphics.
    This is used thousands of times by the AI to learn.
    """
    # Create empty grid
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    # Scatter bricks randomly
    random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
    grid[random_matrix < DENSITY_BRICKS] = 1 
    total_bricks = np.sum(grid)
    
    # Extract Genes (Probabilities)
    genes_pick = genome[0:5]  # Chance to PICK UP a brick
    genes_drop = genome[5:10] # Chance to DROP a brick
    
    # Randomly place termites
    agent_x = np.random.randint(0, GRID_SIZE, NUM_TERMITES)
    agent_y = np.random.randint(0, GRID_SIZE, NUM_TERMITES)
    agent_carrying = np.zeros(NUM_TERMITES, dtype=bool) 
    
    # Run the loop (Training steps)
    for _ in range(STEPS_TRAINING):
        # Move all agents randomly
        shift_x = np.random.choice([-1, 0, 1], NUM_TERMITES)
        shift_y = np.random.choice([-1, 0, 1], NUM_TERMITES)
        agent_x = (agent_x + shift_x) % GRID_SIZE 
        agent_y = (agent_y + shift_y) % GRID_SIZE
        
        # Agents decide what to do
        for i in range(NUM_TERMITES):
            x, y = agent_x[i], agent_y[i]
            
            # Count how many bricks are around this agent (0 to 4)
            n_cnt = 0
            if grid[(x+1)%GRID_SIZE, y] == 1: n_cnt += 1
            if grid[(x-1)%GRID_SIZE, y] == 1: n_cnt += 1
            if grid[x, (y+1)%GRID_SIZE] == 1: n_cnt += 1
            if grid[x, (y-1)%GRID_SIZE] == 1: n_cnt += 1
            here = grid[x, y]
            
            # LOGIC: Should I pick up?
            if not agent_carrying[i] and here == 1:
                # Use the gene corresponding to neighbor count (n_cnt)
                if random.random() < genes_pick[n_cnt]:
                    agent_carrying[i] = True 
                    grid[x, y] = 0           
            # LOGIC: Should I drop?
            elif agent_carrying[i] and here == 0:
                if random.random() < genes_drop[n_cnt]:
                    agent_carrying[i] = False 
                    grid[x, y] = 1            
                    
    # Return the final score of this attempt
    return calculate_order_fast(grid, total_bricks)

def create_genome():
    # Creates a random set of 10 probabilities (0.0 to 1.0)
    return [random.random() for _ in range(10)]

def mutate(genome):
    # Small random change to one gene (Evolution)
    idx = random.randint(0, 9)          
    change = random.uniform(-0.2, 0.2)  
    genome[idx] = max(0.0, min(1.0, genome[idx] + change))
    return genome

def format_genome_str(genome):
    # Helper to make the text pretty for the display
    p = "[" + ", ".join([f"{g:.2f}" for g in genome[0:5]]) + "]"
    d = "[" + ", ".join([f"{g:.2f}" for g in genome[5:10]]) + "]"
    return p, d

# ==========================================
# 4. GRAPHICS SETUP (Matplotlib)
# ==========================================
# Create the main window
fig = plt.figure(figsize=(16, 8)) 

# Create 4 sub-panels (2 for Evo, 2 for Sim)
# We will toggle their visibility later
ax_evol_text = fig.add_subplot(1, 2, 1) 
ax_evol_graph = fig.add_subplot(1, 2, 2)
ax_sim_grid = fig.add_subplot(1, 2, 1) 
ax_sim_graph = fig.add_subplot(1, 2, 2)

def add_thick_border(ax):
    # Adds a black border around a plot
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(4) 

# Initial Visibility: Show Evolution, Hide Simulation
ax_sim_grid.set_visible(False)
ax_sim_graph.set_visible(False)
ax_evol_text.axis('off') 
ax_evol_graph.grid(True)
ax_evol_graph.set_xlim(1, GENERATIONS)
ax_evol_graph.set_ylim(0, 1)

# INITIALIZE TEXT OBJECT (Fixed NameError)
text_display = ax_evol_text.text(0.05, 0.5, "Initializing...", fontsize=12, va='center', fontfamily='monospace')
line_fitness, = ax_evol_graph.plot([], [], 'b-o', lw=2) 

# Setup Simulation Grid Graphics
grid_sim = np.zeros((GRID_SIZE, GRID_SIZE))
# Colors: 0=White, 1=Red(Brick), 2=Blue(Termite), 3=Green(Termite Carrying)
cmap = plt.matplotlib.colors.ListedColormap(['white', 'firebrick', 'cornflowerblue', 'lime'])
bounds = [0, 1, 2, 3, 4]
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
img = ax_sim_grid.imshow(grid_sim, interpolation='nearest', cmap=cmap, norm=norm)
add_thick_border(ax_sim_grid)

sim_history_steps = []
sim_history_metric = []
line_sim, = ax_sim_graph.plot([], [], lw=2, color='firebrick')
ax_sim_graph.set_xlim(0, TOTAL_STEPS_VISUAL)
ax_sim_graph.set_ylim(0, 1)
ax_sim_graph.grid(True)

# ==========================================
# 5. PRE-CALCULATION (Before Video Starts)
# ==========================================
population = [create_genome() for _ in range(POPULATION_SIZE)]

def evaluate_generation(gen_idx):
    """
    Evaluates the entire population, finds the winner,
    logs it, and updates the display text.
    """
    global population, best_genome_overall, best_score_overall, best_info_overall, evolution_history
    global text_display, line_fitness # We need to update these UI elements
    
    scores = []
    for i, genome in enumerate(population):
        # Run 5 trials per genome to be accurate
        trial_scores = [run_simulation_headless(genome) for _ in range(TRIALS_PER_GENOME)]
        avg_score = sum(trial_scores) / len(trial_scores)
        scores.append((avg_score, genome, i))

    # Sort: Highest score first
    scores.sort(key=lambda x: x[0], reverse=True)
    best_gen_score, best_gen_genome, best_species_idx = scores[0]
    
    # --- LOGGING TO FILE ---
    print(f"   > Gen {gen_idx}: Best Score {best_gen_score:.4f}")
    p_str, d_str = format_genome_str(best_gen_genome)
    
    with open(LOG_FILE_PATH, "a") as f:
        f.write(f"GENERATION {gen_idx}\n")
        f.write(f"Best Score: {best_gen_score:.6f} (Species #{best_species_idx})\n")
        f.write(f"Pick: {p_str}\nDrop: {d_str}\n")
        f.write("-" * 40 + "\n")
    
    # Check if this is a new World Record
    if best_gen_score > best_score_overall:
        best_score_overall = best_gen_score
        best_genome_overall = best_gen_genome
        best_info_overall = f"Gen {gen_idx}"
    
    evolution_history.append(best_gen_score)
    
    # --- UPDATE SCREEN TEXT ---
    display_msg = (
        f"GENERATION {gen_idx}\n"
        f"Current Best Fitness: {best_gen_score:.4f}\n\n"
        f"GENOME PARAMETERS:\n"
        f"Pick: {p_str}\n"
        f"Drop: {d_str}"
    )
    text_display.set_text(display_msg)
    
    # Update Graph
    line_fitness.set_data(range(1, len(evolution_history) + 1), evolution_history)
    
    # Save Snapshot
    fig.canvas.draw()
    filename = os.path.join(OUTPUT_FOLDER, f"evol_gen_{gen_idx}.png")
    plt.savefig(filename)
    
    return scores

# !!! Run Generation 1 NOW so Frame 0 has data !!!
print("Initializing Generation 1...")
scores_gen_1 = evaluate_generation(1)

# ==========================================
# 6. MAIN UPDATE LOOP (Video Logic)
# ==========================================
def update(frame):
    global current_state, current_gen, population, pause_counter, sim_step
    global grid_sim, agents, saved_snapshot_data, scores_gen_1, best_genome_overall
    
    # Debug print every 10 frames
    if frame % 10 == 0:
        print(f"[VIDEO] Frame {frame}/{TOTAL_VIDEO_FRAMES} | State: {current_state}")

    # ----------------------------------------------------
    # PHASE 1: EVOLUTION ANIMATION
    # ----------------------------------------------------
    if current_state == STATE_EVOLVING:
        # Calculate which generation we should display based on frame count
        gen_to_show = 1 + (frame // FRAMES_PER_GEN)
        
        if gen_to_show > GENERATIONS:
            print("Evolution Complete. Transitioning...")
            current_state = STATE_TRANSITION
            return []

        # At the start of every "batch" of frames, calculate the NEXT generation
        if frame > 0 and frame % FRAMES_PER_GEN == 0:
            
            # --- BREEDING STEP ---
            top_count = int(POPULATION_SIZE * 0.2)
            
            # We need to re-score the current population to pick parents
            # (We do this here to ensure fresh randomness)
            prev_scores = []
            for genome in population:
                s = [run_simulation_headless(genome) for _ in range(TRIALS_PER_GENOME)]
                prev_scores.append((sum(s)/len(s), genome))
            prev_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Survivors live on
            survivors = [s[1] for s in prev_scores[:top_count]]
            next_gen = survivors[:]
            
            # Fill the rest with mutated children
            while len(next_gen) < POPULATION_SIZE:
                parent = random.choice(survivors)
                child = mutate(copy.deepcopy(parent))
                next_gen.append(child)
            
            population = next_gen
            
            # Evaluate this new generation
            current_gen = gen_to_show
            evaluate_generation(current_gen)
            
        return [line_fitness, text_display]

    # ----------------------------------------------------
    # PHASE 2: TRANSITION (Switch Screens)
    # ----------------------------------------------------
    elif current_state == STATE_TRANSITION:
        # Hide Evo, Show Sim
        ax_evol_text.set_visible(False)
        ax_evol_graph.set_visible(False)
        ax_sim_grid.set_visible(True)
        ax_sim_graph.set_visible(True)
        
        # Initialize Simulation Grid
        random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
        grid_sim = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        grid_sim[random_matrix < DENSITY_BRICKS] = 1
        
        # Initialize Agents
        agents = []
        for _ in range(NUM_TERMITES):
            t = type('', (), {})()
            t.x = random.randint(0, GRID_SIZE-1)
            t.y = random.randint(0, GRID_SIZE-1)
            t.carrying = False
            agents.append(t)
            
        current_state = STATE_PAUSE_START
        pause_counter = 0
        sim_step = 0
        
        # --- CAPTURE STEP 0 SNAPSHOT ---
        # Draw Agents onto grid for the snapshot
        display = grid_sim.copy()
        for t in agents: display[t.x, t.y] = 3 if t.carrying else 2
        img.set_data(display)

        metric = calculate_order_fast(grid_sim, np.sum(grid_sim))
        p_str, d_str = format_genome_str(best_genome_overall)
        header = (f"STEP 0 | Metric: {metric:.3f}\nP: {p_str}\nD: {d_str}")
        ax_sim_grid.set_title(header, fontsize=9, fontweight='bold', loc='left')
        
        filename = os.path.join(OUTPUT_FOLDER, "sim_snap_0.png")
        plt.savefig(filename) 
        saved_snapshot_data[0] = (display.copy(), f"{metric:.3f}", header)
        
        return [img]

    # ----------------------------------------------------
    # PHASE 3: PAUSE (Hold Step 0)
    # ----------------------------------------------------
    elif current_state == STATE_PAUSE_START:
        pause_counter += 1
        if pause_counter >= FRAMES_PAUSE_START:
            current_state = STATE_SIMULATING
        return [img]

    # ----------------------------------------------------
    # PHASE 4: SIMULATION RUN
    # ----------------------------------------------------
    elif current_state == STATE_SIMULATING:
        if sim_step > TOTAL_STEPS_VISUAL:
            current_state = STATE_COMPOSITE
            pause_counter = 0
            return []

        genes_pick = best_genome_overall[0:5]
        genes_drop = best_genome_overall[5:10]
        
        # Run multiple math steps per 1 video frame for speed
        for _ in range(STEPS_PER_FRAME):
            sim_step += 1
            
            # --- SNAPSHOT LOGIC ---
            # Added 10000 to this list as requested
            if sim_step in [2500, 5000, 7500, 10000]:
                metric = calculate_order_fast(grid_sim, np.sum(grid_sim))
                p_str, d_str = format_genome_str(best_genome_overall)
                
                # Update Title specifically for the snapshot
                header = (f"STEP {sim_step} | Metric: {metric:.3f}\nP: {p_str}\nD: {d_str}")
                ax_sim_grid.set_title(header, fontsize=9, fontweight='bold', loc='left')
                
                # IMPORTANT: Draw agents for the snapshot
                # Create a temporary view combining grid + agents
                temp_view = grid_sim.copy()
                for t in agents: temp_view[t.x, t.y] = 3 if t.carrying else 2
                
                # Update the image object temporarily to save correct colors
                img.set_data(temp_view)
                
                # Save
                filename = os.path.join(OUTPUT_FOLDER, f"sim_snap_{sim_step}.png")
                plt.savefig(filename)
                
                # Store for final composite
                saved_snapshot_data[sim_step] = (temp_view.copy(), f"{metric:.3f}", header)
                print(f"     [SNAPSHOT SAVED] {filename}")

            # --- AGENT PHYSICS ---
            for t in agents:
                t.x = (t.x + random.choice([-1, 0, 1])) % GRID_SIZE
                t.y = (t.y + random.choice([-1, 0, 1])) % GRID_SIZE
                
                n_cnt = 0
                if grid_sim[(t.x+1)%GRID_SIZE, t.y] == 1: n_cnt += 1
                if grid_sim[(t.x-1)%GRID_SIZE, t.y] == 1: n_cnt += 1
                if grid_sim[t.x, (t.y+1)%GRID_SIZE] == 1: n_cnt += 1
                if grid_sim[t.x, (t.y-1)%GRID_SIZE] == 1: n_cnt += 1
                here = grid_sim[t.x, t.y]
                
                if not t.carrying and here == 1:
                    if random.random() < genes_pick[n_cnt]:
                        t.carrying = True
                        grid_sim[t.x, t.y] = 0
                elif t.carrying and here == 0:
                    if random.random() < genes_drop[n_cnt]:
                        t.carrying = False
                        grid_sim[t.x, t.y] = 1

        # Real-time Update for Video
        display = grid_sim.copy()
        for t in agents: display[t.x, t.y] = 3 if t.carrying else 2
        img.set_data(display)
        
        curr_metric = calculate_order_fast(grid_sim, np.sum(grid_sim))
        ax_sim_grid.set_title(f"Simulation Step: {sim_step} | Metric: {curr_metric:.3f}", fontsize=12)
        
        sim_history_steps.append(sim_step)
        sim_history_metric.append(curr_metric)
        line_sim.set_data(sim_history_steps, sim_history_metric)
        
        return [img, line_sim]

    # ----------------------------------------------------
    # PHASE 5: FINAL COMPOSITE (The Summary Grid)
    # ----------------------------------------------------
    elif current_state == STATE_COMPOSITE:
        if pause_counter == 0:
            print("Creating Final Composite View...")
            fig.clf() # Clear the whole window
            
            # Create a 2x3 Grid to fit 5 images (0, 2500, 5000, 7500, 10000)
            # subplot(rows, cols, index)
            axes = [fig.add_subplot(2, 3, i+1) for i in range(6)]
            
            steps = [0, 2500, 5000, 7500, 10000]
            
            for i, step in enumerate(steps):
                ax = axes[i]
                if step in saved_snapshot_data:
                    data, metric_s, full_title = saved_snapshot_data[step]
                    
                    cmap = plt.matplotlib.colors.ListedColormap(['white', 'firebrick', 'cornflowerblue', 'lime'])
                    bounds = [0, 1, 2, 3, 4]
                    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
                    ax.imshow(data, interpolation='nearest', cmap=cmap, norm=norm)
                    
                    ax.set_title(full_title, fontsize=6, fontweight='bold')
                    add_thick_border(ax)
                else:
                    ax.text(0.5, 0.5, "Data Missing", ha='center')
                    ax.axis('off')
            
            # Hide the 6th empty slot
            axes[5].axis('off')

            plt.tight_layout()
            
            final_path = os.path.join(OUTPUT_FOLDER, "final_composite_view.png")
            plt.savefig(final_path)
            print(f"     [COMPOSITE SAVED] {final_path}")
            
        pause_counter += 1
        if pause_counter > FRAMES_FINAL_COMPOSITE:
            current_state = STATE_DONE
            
        return []

    return []

# ==========================================
# 7. EXECUTION START
# ==========================================
if __name__ == "__main__":
    ani = animation.FuncAnimation(
        fig, update, 
        frames=TOTAL_VIDEO_FRAMES, 
        interval=1, blit=False, repeat=False
    )
    
    if SAVE_VIDEO:
        video_path = os.path.join(OUTPUT_FOLDER, VIDEO_FILENAME)
        print(f"---------------------------------------")
        print(f"STARTING VIDEO RENDER")
        print(f"Frames: {TOTAL_VIDEO_FRAMES} | FPS: {VIDEO_FPS}")
        print(f"---------------------------------------")
        
        try:
            ani.save(video_path, writer='ffmpeg', fps=VIDEO_FPS)
            print(f"\nSUCCESS: Video saved to {video_path}")
        except Exception as e:
            print(f"\nWARNING: MP4 export failed. Switching to GIF.")
            print(f"Error: {e}")
            gif_path = video_path.replace(".mp4", ".gif")
            ani.save(gif_path, writer='pillow', fps=VIDEO_FPS)
            print(f"SUCCESS: GIF saved to {gif_path}")
    else:
        plt.show()