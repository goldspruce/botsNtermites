import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import random
import copy
import os

# ==========================================
# 1. EXPERIMENTAL SETTINGS
# ==========================================
GRID_SIZE = 50          
NUM_TERMITES = 100      
DENSITY_BRICKS = 0.3    

POPULATION_SIZE = 20    
GENERATIONS = 20        
STEPS_TRAINING = 4000   
TRIALS_PER_GENOME = 5   

TOTAL_STEPS_VISUAL = 10000 
STEPS_PER_FRAME = 200      
SNAPSHOT_STEPS = [0, 2500, 5000, 10000] 

# ==========================================
# 2. GLOBAL VARIABLES & STATE MACHINE
# ==========================================
STATE_EVOLVING = 0     
STATE_WAIT_FOR_SIM = 1 # New state: Wait for user before running best species
STATE_TRANSITION = 2   
STATE_SIMULATING = 3   
STATE_SLIDESHOW = 4  

current_state = STATE_EVOLVING
current_gen = 0
slide_index = 0      
slide_drawn = False  
wait_text_artist = None

best_genome_overall = None     
best_score_overall = 0.0       
best_info_overall = "N/A"         
best_gen_score = 0.0
best_gen_info = "N/A"

saved_evol_snapshots = [] 
saved_sim_snapshots = []
saved_sim_metrics = [] 

# ==========================================
# 3. MECHANISMS
# ==========================================
def calculate_order_fast(grid, total_bricks):
    if total_bricks == 0: return 0
    neighbors = (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                 np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))
    matches = np.sum(neighbors * grid)
    return matches / (total_bricks * 4)

def run_simulation_headless(genome):
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
    grid[random_matrix < DENSITY_BRICKS] = 1 
    total_bricks = np.sum(grid)
    
    genes_pick = genome[0:5]  
    genes_drop = genome[5:10] 
    
    agent_x = np.random.randint(0, GRID_SIZE, NUM_TERMITES)
    agent_y = np.random.randint(0, GRID_SIZE, NUM_TERMITES)
    agent_carrying = np.zeros(NUM_TERMITES, dtype=bool) 
    
    for _ in range(STEPS_TRAINING):
        shift_x = np.random.choice([-1, 0, 1], NUM_TERMITES)
        shift_y = np.random.choice([-1, 0, 1], NUM_TERMITES)
        agent_x = (agent_x + shift_x) % GRID_SIZE 
        agent_y = (agent_y + shift_y) % GRID_SIZE
        
        for i in range(NUM_TERMITES):
            x, y = agent_x[i], agent_y[i]
            here = grid[x, y]
            n_cnt = 0
            if grid[(x+1)%GRID_SIZE, y] == 1: n_cnt += 1
            if grid[(x-1)%GRID_SIZE, y] == 1: n_cnt += 1
            if grid[x, (y+1)%GRID_SIZE] == 1: n_cnt += 1
            if grid[x, (y-1)%GRID_SIZE] == 1: n_cnt += 1
            
            if not agent_carrying[i] and here == 1:
                if random.random() < genes_pick[n_cnt]:
                    agent_carrying[i] = True 
                    grid[x, y] = 0           
            elif agent_carrying[i] and here == 0:
                if random.random() < genes_drop[n_cnt]:
                    agent_carrying[i] = False 
                    grid[x, y] = 1            
                    
    return calculate_order_fast(grid, total_bricks)

# ==========================================
# 4. GENETICS
# ==========================================
def create_genome():
    return [random.random() for _ in range(10)]

def mutate(genome):
    idx = random.randint(0, 9)          
    change = random.uniform(-0.2, 0.2)  
    genome[idx] = max(0.0, min(1.0, genome[idx] + change))
    return genome

population = [create_genome() for _ in range(POPULATION_SIZE)]
evolution_history = [] 

# ==========================================
# 5. GRAPHICS SETUP
# ==========================================
fig = plt.figure(figsize=(16, 8)) 

# Event Listener for the ENTER key
def on_key(event):
    global slide_index, slide_drawn, current_state
    
    # CASE A: Evolution Finished, Waiting to start Simulation
    if current_state == STATE_WAIT_FOR_SIM and event.key == 'enter':
        current_state = STATE_TRANSITION
        print("Starting Best Species Simulation...")
    
    # CASE B: Simulation Finished, User navigating Slideshow
    elif current_state == STATE_SLIDESHOW and event.key == 'enter':
        slide_index += 1
        slide_drawn = False 
        print(f"User pressed ENTER. Showing Slide {slide_index}")

fig.canvas.mpl_connect('key_press_event', on_key)

# --- EVOLUTION DASHBOARD ---
ax_evol_text = fig.add_subplot(1, 2, 1) # Left
ax_evol_graph = fig.add_subplot(1, 2, 2) # Right

ax_evol_text.axis('off') 
ax_evol_text.set_title("Evolutionary Laboratory", fontsize=16, fontweight='bold')
text_display = ax_evol_text.text(0.05, 0.5, "Initializing...", fontsize=11, va='center', fontfamily='monospace')

ax_evol_graph.set_title("Fitness Over Generations")
ax_evol_graph.set_xlabel("Generation")
ax_evol_graph.set_ylabel("Order Metric (Avg)")
ax_evol_graph.set_xlim(1, GENERATIONS) 
ax_evol_graph.set_xticks(np.arange(0, GENERATIONS + 1, 5)) 
ax_evol_graph.set_ylim(0.3, 0.8) 
ax_evol_graph.grid(True) 
line_fitness, = ax_evol_graph.plot([], [], 'b-o', lw=2) 

# Placeholders
ax_sim_grid = None
ax_sim_graph = None
img = None
line_sim = None
grid_sim = None
agents = []
sim_step = 0
sim_history_steps = []
sim_history_metric = []

# Helper function for borders
def add_black_border(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(3) 

# ==========================================
# 6. UPDATE LOOP
# ==========================================

def start_simulation_phase():
    """Setup visual demo"""
    fig.clf() 
    global ax_sim_grid, ax_sim_graph, img, line_sim, grid_sim, agents 
    global sim_history_steps, sim_history_metric, saved_sim_metrics
    
    ax_sim_grid = fig.add_subplot(1, 2, 1)
    ax_sim_graph = fig.add_subplot(1, 2, 2)
    
    # Init World
    grid_sim = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    random_matrix = np.random.rand(GRID_SIZE, GRID_SIZE)
    grid_sim[random_matrix < DENSITY_BRICKS] = 1
    total_bricks = np.sum(grid_sim)
    
    # Step 0 Metric
    start_score = calculate_order_fast(grid_sim, total_bricks)
    sim_history_steps = [0]
    sim_history_metric = [start_score]
    saved_sim_metrics = [start_score] 
    
    agents = []
    for _ in range(NUM_TERMITES):
        t = type('', (), {})() 
        t.x = random.randint(0, GRID_SIZE-1)
        t.y = random.randint(0, GRID_SIZE-1)
        t.carrying = False
        agents.append(t)
    
    cmap = plt.matplotlib.colors.ListedColormap(['white', 'firebrick', 'cornflowerblue', 'lime'])
    bounds = [0, 1, 2, 3, 4]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    img = ax_sim_grid.imshow(grid_sim, interpolation='nearest', cmap=cmap, norm=norm)
    add_black_border(ax_sim_grid) # Border added here
    
    genes_pick = [f"{x:.2f}" for x in best_genome_overall[0:5]]
    genes_drop = [f"{x:.2f}" for x in best_genome_overall[5:10]]
    caption_text = (
        f"BEST GENOME SO FAR: {best_info_overall}\n" 
        f"Fitness: {best_score_overall:.4f}\n"
        f"Pick: {genes_pick}\nDrop: {genes_drop}"
    )
    ax_sim_grid.set_title(caption_text, fontsize=10, fontfamily='monospace', loc='left')
    
    line_sim, = ax_sim_graph.plot(sim_history_steps, sim_history_metric, lw=2, color='firebrick')
    ax_sim_graph.set_xlim(0, TOTAL_STEPS_VISUAL)
    ax_sim_graph.set_ylim(0.3, 0.85) 
    ax_sim_graph.set_title("Real-Time Construction Order")
    ax_sim_graph.grid(True)
    
    # Save the initial Step 0 state
    filename = "sim_snap_0.png"
    extent = ax_sim_grid.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(filename, bbox_inches=extent.expanded(1.05, 1.05)) # Expanded for border
    saved_sim_snapshots.append(filename)

def update(frame):
    global current_state, current_gen, population, best_genome_overall, best_score_overall
    global best_info_overall, best_gen_score, best_gen_info, sim_step, slide_drawn, wait_text_artist
    
    # -----------------------------
    # PHASE 1: EVOLUTION
    # -----------------------------
    if current_state == STATE_EVOLVING:
        if current_gen < GENERATIONS:
            scores = []
            for i, genome in enumerate(population):
                trial_scores = []
                for _ in range(TRIALS_PER_GENOME): 
                    trial_scores.append(run_simulation_headless(genome))
                avg_score = sum(trial_scores) / len(trial_scores)
                scores.append((avg_score, genome, i))

            scores.sort(key=lambda x: x[0], reverse=True)
            best_gen_score, best_gen_genome, best_species_idx = scores[0]
            best_gen_info = f"Gen {current_gen + 1}, Species #{best_species_idx}"
            
            if best_gen_score > best_score_overall:
                best_score_overall = best_gen_score
                best_genome_overall = best_gen_genome
                best_info_overall = best_gen_info
            
            evolution_history.append(best_gen_score)
            
            curr_genes_pick = [f"{x:.2f}" for x in best_gen_genome[0:5]]
            curr_genes_drop = [f"{x:.2f}" for x in best_gen_genome[5:10]]
            
            all_genes_pick = "N/A"
            all_genes_drop = "N/A"
            if best_genome_overall is not None:
                all_genes_pick = [f"{x:.2f}" for x in best_genome_overall[0:5]]
                all_genes_drop = [f"{x:.2f}" for x in best_genome_overall[5:10]]
            
            status_text = (
                f"EVOLUTIONARY STATUS\n"
                f"-------------------\n"
                f"Generation: {current_gen + 1} / {GENERATIONS}\n\n"
                f"BEST GENOME SO FAR:\n"
                f"  {best_info_overall}\n"
                f"  Fitness: {best_score_overall:.4f}\n"
                f"  P: {all_genes_pick}\n"
                f"  D: {all_genes_drop}\n\n"
                f"THIS GENERATION'S BEST GENOME:\n"
                f"  {best_gen_info}\n"
                f"  Fitness: {best_gen_score:.4f}\n"
                f"  P: {curr_genes_pick}\n"
                f"  D: {curr_genes_drop}"
            )
            text_display.set_text(status_text)
            
            gens = range(1, len(evolution_history) + 1)
            line_fitness.set_data(gens, evolution_history)

            # Snapshot Logic
            check_gen = current_gen + 1
            if check_gen in [5, 10, 15, 20]:
                filename = f"evol_snapshot_gen_{check_gen}.png"
                plt.savefig(filename)
                saved_evol_snapshots.append(filename)
            
            top_count = int(POPULATION_SIZE * 0.2)
            survivors = [s[1] for s in scores[:top_count]]
            next_gen = survivors[:]
            while len(next_gen) < POPULATION_SIZE:
                parent = random.choice(survivors)
                child = mutate(copy.deepcopy(parent))
                next_gen.append(child)
            
            population = next_gen 
            current_gen += 1      
            return [line_fitness, text_display]
        else:
            # Move to waiting state instead of immediate transition
            current_state = STATE_WAIT_FOR_SIM
            
            # Show "Press Enter" message
            if wait_text_artist is None:
                wait_text_artist = fig.text(0.5, 0.5, "EVOLUTION COMPLETE\nPress [ENTER] to visualize best species", 
                                            ha='center', va='center', fontsize=20, 
                                            color='white', bbox=dict(facecolor='black', alpha=0.8, edgecolor='red'))
            return [wait_text_artist]

    # -----------------------------
    # PHASE 1.5: WAITING FOR USER
    # -----------------------------
    elif current_state == STATE_WAIT_FOR_SIM:
        # Just idle until Key Press event changes state
        return []

    # -----------------------------
    # PHASE 2: TRANSITION
    # -----------------------------
    elif current_state == STATE_TRANSITION:
        # Clear the "Press Enter" message if it exists
        if wait_text_artist:
            wait_text_artist.remove()
            wait_text_artist = None
            
        start_simulation_phase() 
        current_state = STATE_SIMULATING
        return [img, line_sim]

    # -----------------------------
    # PHASE 3: SIMULATION
    # -----------------------------
    elif current_state == STATE_SIMULATING:
        if sim_step > TOTAL_STEPS_VISUAL:
            current_state = STATE_SLIDESHOW
            slide_drawn = False 
            return []
            
        genes_pick = best_genome_overall[0:5]
        genes_drop = best_genome_overall[5:10]
        
        for _ in range(STEPS_PER_FRAME):
            if sim_step > TOTAL_STEPS_VISUAL: break
            sim_step += 1
            
            # --- CAPTURE SIMULATION STEPS ---
            if sim_step in [2500, 5000, 10000]:
                current_metric = calculate_order_fast(grid_sim, np.sum(grid_sim))
                saved_sim_metrics.append(current_metric)
                
                filename = f"sim_snap_{sim_step}.png"
                extent = ax_sim_grid.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(filename, bbox_inches=extent.expanded(1.05, 1.05)) 
                saved_sim_snapshots.append(filename)

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
        
        display = grid_sim.copy()
        for t in agents: display[t.x, t.y] = 3 if t.carrying else 2
        img.set_data(display)
        
        current_metric = calculate_order_fast(grid_sim, np.sum(grid_sim))
        sim_history_steps.append(sim_step)
        sim_history_metric.append(current_metric)
        line_sim.set_data(sim_history_steps, sim_history_metric)
        return [img, line_sim]

    # -----------------------------
    # PHASE 4: INTERACTIVE SLIDESHOW
    # -----------------------------
    elif current_state == STATE_SLIDESHOW:
        if not slide_drawn:
            fig.clf()
            
            # SLIDES 0-3: Evolution Generations (One by one)
            if slide_index < 4:
                gen_nums = [5, 10, 15, 20]
                fig.suptitle(f"Evolution Phase: Generation {gen_nums[slide_index]}\nPress [ENTER] for next", fontsize=16, color='blue')
                if len(saved_evol_snapshots) > slide_index:
                    ax = fig.add_subplot(1, 1, 1)
                    ax.imshow(mpimg.imread(saved_evol_snapshots[slide_index]))
                    ax.axis('off')

            # SLIDE 4: Final Construction Sequence
            else:
                # REMOVED top caption entirely as requested
                # fig.suptitle(...) <--- Gone
                
                gs = fig.add_gridspec(1, 4, wspace=0.1, top=0.85, bottom=0.1, left=0.05, right=0.95)
                
                for i, filename in enumerate(saved_sim_snapshots):
                    if i >= 4: break
                    ax = fig.add_subplot(gs[i])
                    ax.imshow(mpimg.imread(filename))
                    
                    # CHANGED Caption to only "Order Metric: X"
                    metric_val = saved_sim_metrics[i] if i < len(saved_sim_metrics) else 0.0
                    ax.set_title(f"Order Metric: {metric_val:.4f}", fontsize=11)
                    
                    add_black_border(ax)

            plt.draw()
            slide_drawn = True 
            
        return []

    return []

if __name__ == "__main__":
    print("------------------------------------------------")
    print("Running Thesis Simulation...")
    print("NOTE: Evolution will run first.")
    print("      Then, press ENTER to start the simulation.")
    print("      Then, press ENTER to advance through results.")
    print("------------------------------------------------")
    ani = animation.FuncAnimation(fig, update, interval=1, blit=False, repeat=False, cache_frame_data=False)
    plt.show()
