# ==========================================
# REPLACE SECTION 3 & 5 WITH THIS UPDATED CODE
# ==========================================

import multiprocessing # <--- NEW IMPORT

# --- UPDATED HELPER FUNCTIONS (SECTION 3) ---

def calculate_order_fast(grid, total_bricks):
    if total_bricks == 0: return 0
    neighbors = (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                 np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))
    matches = np.sum(neighbors * grid)
    return matches / (total_bricks * 4)

def run_simulation_headless(genome):
    # (Identical to your original function)
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
            
            n_cnt = 0
            if grid[(x+1)%GRID_SIZE, y] == 1: n_cnt += 1
            if grid[(x-1)%GRID_SIZE, y] == 1: n_cnt += 1
            if grid[x, (y+1)%GRID_SIZE] == 1: n_cnt += 1
            if grid[x, (y-1)%GRID_SIZE] == 1: n_cnt += 1
            here = grid[x, y]
            
            if not agent_carrying[i] and here == 1:
                if random.random() < genes_pick[n_cnt]:
                    agent_carrying[i] = True 
                    grid[x, y] = 0           
            elif agent_carrying[i] and here == 0:
                if random.random() < genes_drop[n_cnt]:
                    agent_carrying[i] = False 
                    grid[x, y] = 1            
                    
    return calculate_order_fast(grid, total_bricks)

# --- NEW WORKER FUNCTION FOR MULTIPROCESSING ---
def worker_evaluation_task(genome):
    """
    This function runs on a separate CPU core.
    It runs all 5 trials for a single genome and returns the average.
    """
    # Re-seed random number generator to ensure unique results per core
    np.random.seed() 
    random.seed()
    
    scores = [run_simulation_headless(genome) for _ in range(TRIALS_PER_GENOME)]
    avg_score = sum(scores) / len(scores)
    return avg_score

def create_genome():
    return [random.random() for _ in range(10)]

def mutate(genome):
    idx = random.randint(0, 9)          
    change = random.uniform(-0.2, 0.2)  
    genome[idx] = max(0.0, min(1.0, genome[idx] + change))
    return genome

def format_genome_str(genome):
    p = "[" + ", ".join([f"{g:.2f}" for g in genome[0:5]]) + "]"
    d = "[" + ", ".join([f"{g:.2f}" for g in genome[5:10]]) + "]"
    return p, d

# --- UPDATED EVALUATION LOGIC (SECTION 5) ---
population = [create_genome() for _ in range(POPULATION_SIZE)]

def evaluate_generation(gen_idx):
    global population, best_genome_overall, best_score_overall, best_info_overall, evolution_history
    global text_display, line_fitness

    # --- PARALLEL PROCESSING START ---
    # We create a Pool of workers equal to the number of CPU cores
    # 'map' automatically distributes the 'population' list across cores
    with multiprocessing.Pool() as pool:
        # returns a list of scores corresponding to the population
        avg_scores = pool.map(worker_evaluation_task, population)
    # --- PARALLEL PROCESSING END ---

    # Combine scores with genomes so we can sort them
    scores = []
    for i, score in enumerate(avg_scores):
        scores.append((score, population[i], i))

    # The rest of the function remains exactly the same...
    scores.sort(key=lambda x: x[0], reverse=True)
    best_gen_score, best_gen_genome, best_species_idx = scores[0]
    
    print(f"   > Gen {gen_idx}: Best Score {best_gen_score:.4f}")
    p_str, d_str = format_genome_str(best_gen_genome)
    
    with open(LOG_FILE_PATH, "a") as f:
        f.write(f"GENERATION {gen_idx}\n")
        f.write(f"Best Score: {best_gen_score:.6f} (Species #{best_species_idx})\n")
        f.write(f"Pick: {p_str}\nDrop: {d_str}\n")
        f.write("-" * 40 + "\n")
    
    if best_gen_score > best_score_overall:
        best_score_overall = best_gen_score
        best_genome_overall = best_gen_genome
        best_info_overall = f"Gen {gen_idx}"
    
    evolution_history.append(best_gen_score)
    
    display_msg = (
        f"GENERATION {gen_idx}\n"
        f"Current Best Fitness: {best_gen_score:.4f}\n\n"
        f"GENOME PARAMETERS:\n"
        f"Pick: {p_str}\n"
        f"Drop: {d_str}"
    )
    text_display.set_text(display_msg)
    
    line_fitness.set_data(range(1, len(evolution_history) + 1), evolution_history)
    
    fig.canvas.draw()
    filename = os.path.join(OUTPUT_FOLDER, f"evol_gen_{gen_idx}.png")
    plt.savefig(filename)
    
    return scores