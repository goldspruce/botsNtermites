# =============================================================================
# ENV FIX: Attempting to suppress macOS SDL2 conflict warnings
# =============================================================================
import os
import sys
os.environ['PYTHON_APPLE_ALLOW_SDL2_MIX'] = '1'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1' # Cleans up the console a bit more

# =============================================================================
# IMPORTS: Bringing in outside toolboxes so we don't have to write everything from scratch.
# =============================================================================
import pygame       # Pygame is a library for making 2D games and drawing graphics.
import math         # Gives us math tools, like trigonometry (sin/cos) and calculating distances.
import random       # Lets us generate random numbers so the bbots don't always do the exact same thing.
import csv          # Lets us read and write data to Comma Separated Value (.csv) spreadsheet files.
import statistics   # Gives us statistical math tools, like calculating the Mean (average) and Standard Deviation.
from datetime import datetime   # Lets us pull the current time from your computer clock to name our folders.
import matplotlib.pyplot as plt # A popular library for drawing graphs and charts (used for our histograms).

# --- NEW IMPORTS FOR VIDEO EXPORT ---
import cv2          # OpenCV: The gold standard for video writing and image processing.
import numpy as np  # NumPy: Handles the heavy matrix math needed to convert screen pixels to a video file.

# =============================================================================
# SECTION 1: GLOBAL CONFIGURATION (CONSTANTS)
# =============================================================================

# --- Screen & Display Settings ---
WIDTH, HEIGHT = 600, 600          # The size of our simulation window in pixels.

# --- Colors (RGB format: Red, Green, Blue) ---
BG_COLOR = (20, 20, 20)           # Dark gray background.
LIGHT_COLOR = (101, 67, 33)       # Dark brown to denote an inactive light/darkness.
BBOT_COLOR = (220, 60, 60)        # Red color for moving bbots.
REST_COLOR = (60, 220, 60)        # Green color for bbots that have stopped.
BLOCK_COLOR = (80, 80, 220)       # Blue color for the pushable blocks (rocks).
WALL_COLOR = (255, 255, 0)        # Yellow color for the boundary walls.
TEXT_COLOR = (255, 255, 255)      # White color for on-screen text.

# --- Light Source Parameters ---
LIGHT_POS = (15, 15)              # Coordinates (x, y) for the center of the light source.
LIGHT_RADIUS_VISUAL = 5           # How big the light bulb looks on screen.

# --- Entity Counts & Sizes ---
NUM_BBOTS = 50                    # Total number of bbots in the arena.
NUM_BLOCKS = 20                   # Total number of pushable blocks in the arena.
BLOCK_RADIUS = 20                 # Physical size of the blocks.
BBOT_RADIUS = 10                  # Physical size of the bbots.
WALL_THICKNESS = 10               # How thick the boundary walls are.

# --- Physics & Movement Settings ---
BBOT_SPEED_GO = 3.5               # How fast bbots move when driving freely.
BBOT_SPEED_PUSH = 1.0             # How fast bbots move when pushing a block.
PUSH_FORCE = 1.0                  # How much force a bbot applies to a block when they collide.
BLOCK_FRICTION = 0.90             # How quickly a block slows down after being pushed (1.0 = no friction).

# --- Simulation Rules ---
SIM_TIMEOUT = 300.0               # Maximum time (in seconds) the simulation will run before forcing a stop.

# =============================================================================
# SECTION 2: HELPER FUNCTIONS & CLASSES
# =============================================================================

def save_screenshot(screen, folder, filename):
    """Saves the current Pygame screen as an image file."""
    path = os.path.join(folder, filename) 
    pygame.image.save(screen, path)

def get_bbot_spawn_pos():
    """Calculates a random, safe starting position for a bbot inside the walls."""
    safe_min = WALL_THICKNESS + BBOT_RADIUS
    safe_max_x = WIDTH - WALL_THICKNESS - BBOT_RADIUS
    safe_max_y = HEIGHT - WALL_THICKNESS - BBOT_RADIUS
    x = random.uniform(safe_min, safe_max_x) 
    y = random.uniform(safe_min, safe_max_y) 
    return x, y 

def get_valid_block_pos(existing_bbots):
    """Finds a random starting position for a block, ensuring it doesn't overlap with a bbot."""
    safe_min = WALL_THICKNESS + BLOCK_RADIUS
    safe_max_x = WIDTH - WALL_THICKNESS - BLOCK_RADIUS
    safe_max_y = HEIGHT - WALL_THICKNESS - BLOCK_RADIUS
    while True: 
        x = random.randint(int(safe_min), int(safe_max_x))
        y = random.randint(int(safe_min), int(safe_max_y))
        
        # Check if this spot is too close to any existing bbot
        if any(math.hypot(x - b.x, y - b.y) < (BLOCK_RADIUS + BBOT_RADIUS + 2) for b in existing_bbots):
            continue 
        return x, y 

def get_distance_pixels(x, y):
    """Calculates the straight-line distance from a given point to the light source."""
    return math.hypot(x - LIGHT_POS[0], y - LIGHT_POS[1])

def draw_text(screen, text, size, x, y, color=TEXT_COLOR):
    """A helper to easily draw text on the Pygame screen."""
    font = pygame.font.SysFont("Arial", size, bold=False)
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y)) 

# --- CLASSES ---

class BBot:
    """Represents a single tumbling robot in the simulation."""
    def __init__(self, target_mean_steps, target_sd_steps):
        self.x, self.y = get_bbot_spawn_pos() 
        self.start_x, self.start_y = self.x, self.y
        self.angle = random.uniform(0, 2 * math.pi) 
        
        self.touching_block = False 
        self.is_stopped = False
        self.stop_tick = None 
        self.stop_time = None
        
        # Stochastic Stoppage: Truncated Normal Distribution based on STEPS
        while True:
            t = random.gauss(target_mean_steps, target_sd_steps)
            if t > 0: 
                self.target_stop_steps = t
                break

    def decide_and_move(self, current_steps):
        """Controls movement. Cuts motor if its personal tick timer is up."""
        if current_steps >= self.target_stop_steps:
            self.is_stopped = True
            return 
        
        speed = BBOT_SPEED_PUSH if self.touching_block else BBOT_SPEED_GO
        
        if not self.touching_block: 
            self.angle += random.uniform(-0.4, 0.4) 

        self.x += math.cos(self.angle) * speed
        self.y += math.sin(self.angle) * speed
        
        min_bound = WALL_THICKNESS + BBOT_RADIUS
        self.x = max(min_bound, min(WIDTH - min_bound, self.x))
        self.y = max(min_bound, min(HEIGHT - min_bound, self.y))

class Block:
    """Represents a pushable rock/block in the simulation."""
    def __init__(self, existing_bbots):
        self.radius = BLOCK_RADIUS
        self.x, self.y = get_valid_block_pos(existing_bbots)
        self.start_x, self.start_y = self.x, self.y 
        self.vx, self.vy = 0.0, 0.0 

    def update(self):
        """Applies velocity and friction to the block to make it slide smoothly."""
        self.x += self.vx
        self.y += self.vy
        self.vx *= BLOCK_FRICTION
        self.vy *= BLOCK_FRICTION
        
        # Keep block inside boundaries
        min_bound = WALL_THICKNESS + self.radius
        if self.x < min_bound: self.x = min_bound; self.vx = 0
        if self.x > WIDTH - min_bound: self.x = WIDTH - min_bound; self.vx = 0
        if self.y < min_bound: self.y = min_bound; self.vy = 0
        if self.y > HEIGHT - min_bound: self.y = HEIGHT - min_bound; self.vy = 0

def check_collisions(bbots, blocks):
    """Handles the physics of bbots hitting blocks, and blocks hitting other blocks."""
    # 1. BBot vs Block Collisions
    for bbot in bbots:
        bbot.touching_block = False # Reset flag each frame
        for block in blocks:
            dx = block.x - bbot.x
            dy = block.y - bbot.y
            dist = math.hypot(dx, dy)
            # If the distance is less than their combined radii, they are touching
            if dist < (BBOT_RADIUS + block.radius):
                bbot.touching_block = True
                if dist < 0.1: dist = 0.1 # Prevent division by zero
                # Calculate the angle of the push
                nx, ny = dx / dist, dy / dist
                # Transfer force to the block
                block.vx += nx * PUSH_FORCE
                block.vy += ny * PUSH_FORCE
                # Bounce the bbot back slightly
                bbot.x -= nx * 2
                bbot.y -= ny * 2

    # 2. Block vs Block Collisions
    for i, b1 in enumerate(blocks):
        for b2 in blocks[i+1:]: 
            dx = b1.x - b2.x
            dy = b1.y - b2.y
            dist = math.hypot(dx, dy)
            if dist < (b1.radius + b2.radius):
                if dist < 0.1: dist = 0.1
                nx, ny = dx / dist, dy / dist
                # Push both blocks away from each other
                b1.x += nx * 2.0; b1.y += ny * 2.0
                b2.x -= nx * 2.0; b2.y -= ny * 2.0

# =============================================================================
# SECTION 3: EXPORTING, REPORTING & PLOTTING
# =============================================================================

def export_trial_1_metrics(result, target_mean, target_sd, output_folder):
    """Saves a dedicated CSV for just Trial 1's results without rounding."""
    filename = os.path.join(output_folder, "trial_01_metrics.csv")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["trial_01_metrics.csv"])
        writer.writerow([])
        
        # User Inputs (Steps)
        writer.writerow(["Target Mean Stop Time (steps)", target_mean])
        writer.writerow(["Target SD of Stop Times (steps)", target_sd])
        writer.writerow([])
        
        # Actual Stoppage Times & Duration (No Rounding)
        st_steps = result['stop_ticks']
        st_time = result['stop_times']
        
        m_stop_steps = statistics.mean(st_steps) if st_steps else 0.0
        sd_stop_steps = statistics.stdev(st_steps) if len(st_steps) > 1 else 0.0
        m_stop_time = statistics.mean(st_time) if st_time else 0.0
        sd_stop_time = statistics.stdev(st_time) if len(st_time) > 1 else 0.0
        
        writer.writerow(["Duration (Steps)", result['duration_steps']])
        writer.writerow(["Duration (Video Seconds)", result['duration']])
        writer.writerow(["Actual Mean Stop Time (steps)", m_stop_steps])
        writer.writerow(["Actual SD of Stop Times (steps)", sd_stop_steps])
        writer.writerow(["Actual Mean Stop Time (s)", m_stop_time])
        writer.writerow(["Actual SD of Stop Times (s)", sd_stop_time])
        writer.writerow([])
        
        # Rock Displacements (No Rounding)
        rd = result['rock_displacements']
        sd_rock = statistics.stdev(rd) if len(rd) > 1 else 0.0
        writer.writerow(["Mean Rock Displacement (px)", result['mean_rock_displacement']])
        writer.writerow(["Rock Displacement SD (px)", sd_rock])
        writer.writerow([])
        
        # BBot Displacements (No Rounding)
        bd = result['bbot_displacements']
        sd_bbot = statistics.stdev(bd) if len(bd) > 1 else 0.0
        writer.writerow(["Mean BBot Displacement (px)", result['mean_bbot_displacement']])
        writer.writerow(["BBot Displacement SD (px)", sd_bbot])


def export_aggregate_summary(aggregate_results, output_folder, final_stats, target_mean, target_sd):
    """Saves the master summary incorporating intra-run and pooled statistics (Excluding Trial 1, No Rounding)."""
    filename = os.path.join(output_folder, "aggregate_summary.csv")

    # Flag any anomalous trials where the net displacement was backwards (negative)
    bad_rock_trials = [str(r['trial']) for r in aggregate_results if r['mean_rock_displacement'] <= 0]
    bad_bbot_trials = [str(r['trial']) for r in aggregate_results if r['mean_bbot_displacement'] <= 0]

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(["aggregate_summary.csv"])
        writer.writerow(["*Note: Trial 1 has been explicitly excluded from this file."])
        writer.writerow([])
        
        # Save the user-inputted parameters
        writer.writerow(["Target Mean Stop Time (steps)", target_mean])
        writer.writerow(["Target SD of Stop Times (steps)", target_sd])
        writer.writerow([])
        
        writer.writerow(["Trials with Mean Rock Displacement <= 0", len(bad_rock_trials)])
        writer.writerow(["Specific Rock Trial IDs", ", ".join(bad_rock_trials) if bad_rock_trials else "None"])
        writer.writerow([])
        writer.writerow(["Trials with Mean BBot Displacement <= 0", len(bad_bbot_trials)])
        writer.writerow(["Specific BBot Trial IDs", ", ".join(bad_bbot_trials) if bad_bbot_trials else "None"])
        writer.writerow([])
        
        writer.writerow(["Total Aggregated Trials", len(aggregate_results)])
        writer.writerow([])
        
        # Rock Displacement (No Rounding)
        writer.writerow(["", "Mean of Pooled Rock Displacement (px)", final_stats['m_pool_rock']])
        writer.writerow(["", "SD of Above (px)", final_stats['sd_pool_rock']])
        writer.writerow(["", "Mean of Intra-Run Rock Displacement (px)", final_stats['m_intra_rock']])
        writer.writerow(["", "SD of Above (px)", final_stats['sd_intra_rock']])
        writer.writerow([])
        
        # BBot Displacements (No Rounding)
        writer.writerow(["", "Mean of Pooled Bbot Displacements (px)", final_stats['m_pool_bbot_disp']])
        writer.writerow(["", "SD of Above (px)", final_stats['sd_pool_bbot_disp']])
        writer.writerow(["", "Mean of Intra-Run Bbot Displacements (px)", final_stats['m_intra_bbot_disp']])
        writer.writerow(["", "SD of Above (px)", final_stats['sd_intra_bbot_disp']])
        writer.writerow([])
        
        # BBot Stop Steps (No Rounding)
        writer.writerow(["", "Mean of Pooled Bbot Stop Times (steps)", final_stats['m_pool_steps']])
        writer.writerow(["", "SD of Above (steps)", final_stats['sd_pool_steps']])
        writer.writerow(["", "Mean of Intra-Run Stop Time Means (steps)", final_stats['m_intra_steps']])
        writer.writerow(["", "SD of Above (steps)", final_stats['sd_intra_steps']])
        writer.writerow([])
        
        # BBot Stop Times (Seconds) (No Rounding)
        writer.writerow(["", "Mean of Pooled Bbot Stop Times (s)", final_stats['m_pool_time']])
        writer.writerow(["", "SD of Above (s)", final_stats['sd_pool_time']])
        writer.writerow(["", "Mean of Intra-Run Stop Time Means (s)", final_stats['m_intra_time']])
        writer.writerow(["", "SD of Above (s)", final_stats['sd_intra_time']])
        writer.writerow([])
        writer.writerow([])
        
        # Trial By Trial Breakdown (Trial 1 Excluded, No Rounding)
        writer.writerow([
            "Trial", "Duration (steps)", "Duration (s)", 
            "Mean Rock Displacement", "Rock Displacement SD", 
            "Mean BBot Displacement", "BBot Displacement SD", 
            "Mean BBot Stop Time (steps)", "BBot Stop SD (steps)",
            "Mean BBot Stop Time (s)", "BBot Stop SD (s)"
        ])
        for r in aggregate_results:
            st_steps = r['stop_ticks']
            st_time = r['stop_times']
            rd = r['rock_displacements']
            bd = r['bbot_displacements']
            writer.writerow([
                r['trial'], 
                r['duration_steps'], 
                r['duration'], 
                r['mean_rock_displacement'], 
                statistics.stdev(rd) if len(rd) > 1 else 0.0,
                r['mean_bbot_displacement'],
                statistics.stdev(bd) if len(bd) > 1 else 0.0,
                statistics.mean(st_steps) if st_steps else 0.0,
                statistics.stdev(st_steps) if len(st_steps) > 1 else 0.0,
                statistics.mean(st_time) if st_time else 0.0,
                statistics.stdev(st_time) if len(st_time) > 1 else 0.0
            ])

def plot_histogram(data, title, xlabel, ylabel, filename, color, output_folder, is_time=False):
    """Generates and saves a histogram image for the given dataset."""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=40, color=color, edgecolor='black', alpha=0.7)
    mean_val = statistics.mean(data)
    unit = "s" if is_time else "px"
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}{unit}')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    sd_val = statistics.stdev(data) if len(data) > 1 else 0
    info_text = f"Pooled SD: {sd_val:.2f}{unit}\nTotal Samples: {len(data)}"
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()

# =============================================================================
# SECTION 4: USER INTERFACE ROUTINES
# =============================================================================

def ui_get_simulation_parameters(screen):
    """Sequential menu to get trials, target mean, and target SD."""
    clock = pygame.time.Clock()
    prompts = [
        "1. Enter number of trials:",
        "2. Enter Target Mean Stop Time (steps):",
        "3. Enter Target SD of Stop Times (steps):"
    ]
    inputs = ["", "", ""]
    step = 0

    while step < len(prompts):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                pygame.quit()
                sys.exit() 
                
            if event.type == pygame.KEYDOWN:
                if step >= len(prompts): 
                    break 
                    
                if event.key == pygame.K_RETURN:
                    if inputs[step].strip(): 
                        step += 1
                elif event.key == pygame.K_BACKSPACE:
                    inputs[step] = inputs[step][:-1] 
                elif event.unicode.isnumeric() or event.unicode == '.': 
                    inputs[step] += event.unicode 

        if step < len(prompts):
            screen.fill(BG_COLOR) 
            draw_text(screen, "DARK BBOT CONTROL", 28, WIDTH//2 - 160, HEIGHT//2 - 100, (255, 215, 0))
            draw_text(screen, prompts[step], 22, WIDTH//2 - 200, HEIGHT//2 - 20)
            draw_text(screen, inputs[step] + "_", 36, WIDTH//2 - 30, HEIGHT//2 + 30, (100, 255, 100))
            pygame.display.flip() 
            
        clock.tick(30) 
        
    num_trials = max(1, int(inputs[0])) if inputs[0] else 1
    mean_steps = float(inputs[1]) if inputs[1] else 600.0
    sd_steps = float(inputs[2]) if inputs[2] else 120.0
    
    return num_trials, mean_steps, sd_steps

def ui_show_headless_progress(screen, current, total):
    """Shows a simple progress screen during fast batch simulation."""
    screen.fill(BG_COLOR)
    draw_text(screen, "RUNNING HEADLESS SIMULATIONS", 28, WIDTH//2 - 200, HEIGHT//2 - 40, (255, 215, 0))
    draw_text(screen, f"Crunching Trial {current} / {total}...", 20, WIDTH//2 - 100, HEIGHT//2 + 10)
    pygame.display.flip()
    pygame.event.pump() 

def ui_show_trial1_results(screen, duration_secs, duration_steps, mean_rock_displacement, sd_rock_displacement, mean_bbot_displacement, sd_bbot_displacement, output_folder, mean_stop_time, sd_stop_time, mean_stop_steps, sd_stop_steps, target_mean, target_sd):
    """Displays the stats of Trial 1 before beginning headless processing, comparing targets to actuals."""
    waiting = True
    screenshot_taken = False
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN: waiting = False

        screen.fill(BG_COLOR)
        draw_text(screen, "TRIAL 1 COMPLETE", 32, 30, 20, (255, 215, 0))
        
        # Duration: Steps above Seconds
        draw_text(screen, f"Duration (Steps):      {duration_steps}", 18, 30, 70)
        draw_text(screen, f"Duration (Video Secs): {duration_secs:.2f} s", 18, 30, 95)
        
        # Displaying INPUT (Target) vs ACTUAL
        draw_text(screen, f"Target Mean Stop:      {target_mean:.2f} steps", 18, 30, 135, (100, 255, 100))
        draw_text(screen, f"Target SD of Stops:    {target_sd:.2f} steps", 18, 30, 160, (100, 255, 100))
        
        # Steps Data above Seconds Data
        draw_text(screen, f"Actual Mean Stop:      {mean_stop_steps:.2f} steps", 18, 30, 200)
        draw_text(screen, f"Actual SD of Stops:    {sd_stop_steps:.2f} steps", 18, 30, 225, (200, 200, 200))
        draw_text(screen, f"Actual Mean Stop:      {mean_stop_time:.2f} s", 18, 30, 265)
        draw_text(screen, f"Actual SD of Stops:    {sd_stop_time:.2f} s", 18, 30, 290, (200, 200, 200))
        
        # Displacement Stats (Moved to the right so they don't overlap)
        sign_rock = "" if mean_rock_displacement < 0 else "+"
        draw_text(screen, f"Mean Rock Disp:      {sign_rock}{mean_rock_displacement:.2f} px", 18, 330, 135)
        draw_text(screen, f"Rock Disp SD:        {sd_rock_displacement:.2f} px", 18, 330, 160, (200, 200, 200))
        
        sign_bbot = "" if mean_bbot_displacement < 0 else "+"
        draw_text(screen, f"Mean BBot Disp:      {sign_bbot}{mean_bbot_displacement:.2f} px", 18, 330, 200)
        draw_text(screen, f"BBot Disp SD:        {sd_bbot_displacement:.2f} px", 18, 330, 225, (200, 200, 200))

        draw_text(screen, "Press [ENTER] to begin Headless Runs.", 22, 30, HEIGHT - 50, (255, 100, 100))
        pygame.display.flip()
        
        if not screenshot_taken:
            save_screenshot(screen, output_folder, "04_final_metrics.png")
            screenshot_taken = True 
        pygame.time.Clock().tick(15)

def ui_show_final_done(screen, output_folder):
    """Final screen shown when all data generation is fully complete."""
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN):
                waiting = False
        screen.fill(BG_COLOR)
        draw_text(screen, "BATCH COMPLETE", 32, WIDTH//2 - 120, HEIGHT//2 - 60, (60, 220, 60))
        draw_text(screen, f"Data and Histograms successfully saved to:", 18, WIDTH//2 - 160, HEIGHT//2)
        draw_text(screen, f"{output_folder}", 16, WIDTH//2 - 180, HEIGHT//2 + 30, (200, 200, 200))
        draw_text(screen, "Press [ENTER] to exit.", 18, WIDTH//2 - 80, HEIGHT//2 + 80, (255, 255, 0))
        pygame.display.flip()
        pygame.time.Clock().tick(15)

# =============================================================================
# SECTION 5: SIMULATION RUNNER
# =============================================================================

def run_simulation(trial_num, is_visual, output_folder, target_mean, target_sd, screen=None):
    """Runs a single instance of the simulation."""
    bbots = [BBot(target_mean, target_sd) for _ in range(NUM_BBOTS)]
    blocks = [Block(bbots) for _ in range(NUM_BLOCKS)] 
    
    ticks, sim_time, final_time = 0, 0.0, None
    video_writer = None 
    
    if is_visual and screen:
        pygame.display.set_caption("DARK BBOT CONTROL")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_filename = os.path.join(output_folder, "00_trial_01_video.mp4")
        video_writer = cv2.VideoWriter(video_filename, fourcc, 60.0, (WIDTH, HEIGHT))
        
        clock = pygame.time.Clock()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN: waiting = False
                
            screen.fill(BG_COLOR)
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, HEIGHT-WALL_THICKNESS, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WALL_THICKNESS, HEIGHT))
            pygame.draw.rect(screen, WALL_COLOR, (WIDTH-WALL_THICKNESS, 0, WALL_THICKNESS, HEIGHT))
            pygame.draw.circle(screen, LIGHT_COLOR, LIGHT_POS, LIGHT_RADIUS_VISUAL)
            
            for b in blocks: pygame.draw.circle(screen, BLOCK_COLOR, (int(b.x), int(b.y)), int(b.radius))
            for bbot in bbots: pygame.draw.circle(screen, BBOT_COLOR, (int(bbot.x), int(bbot.y)), BBOT_RADIUS)
            
            draw_text(screen, "Press ENTER to Start Trial 1", 32, WIDTH//2 - 170, 100, (255, 255, 0))
            pygame.display.flip()
            clock.tick(60)
            
        save_screenshot(screen, output_folder, "01_sim_start.png")
        last_shot_time = 0

    # THE MAIN PHYSICS LOOP
    running = True
    while running:
        if is_visual:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()

        ticks += 1
        sim_time = ticks / 60.0  
        
        check_collisions(bbots, blocks) 
        for b in blocks: b.update() 
        
        safe_bbots = 0 
        for bbot in bbots:
            # Tell the bot exactly what step we are on so it can stop if necessary
            bbot.decide_and_move(ticks) 
            
            if bbot.is_stopped:
                safe_bbots += 1
                if bbot.stop_tick is None: 
                    bbot.stop_tick = ticks
                    bbot.stop_time = sim_time 
                
        if safe_bbots == NUM_BBOTS or sim_time >= SIM_TIMEOUT:
            final_time = sim_time
            running = False 
            
        if is_visual and screen:
            screen.fill(BG_COLOR)
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, HEIGHT-WALL_THICKNESS, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WALL_THICKNESS, HEIGHT))
            pygame.draw.rect(screen, WALL_COLOR, (WIDTH-WALL_THICKNESS, 0, WALL_THICKNESS, HEIGHT))
            pygame.draw.circle(screen, LIGHT_COLOR, LIGHT_POS, LIGHT_RADIUS_VISUAL)
            
            # UPDATED: Render both the simulated time and exact step count
            draw_text(screen, f"{sim_time:.2f}s | {ticks} steps", 20, 10, 80, (255, 255, 0))
            
            for b in blocks: pygame.draw.circle(screen, BLOCK_COLOR, (int(b.x), int(b.y)), int(b.radius))
            for bbot in bbots:
                color = REST_COLOR if bbot.is_stopped else BBOT_COLOR
                pygame.draw.circle(screen, color, (int(bbot.x), int(bbot.y)), BBOT_RADIUS)
                
            pygame.display.flip() 
            
            if video_writer is not None:
                frame = pygame.surfarray.array3d(screen)
                frame = frame.transpose([1, 0, 2])
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)
            
            if sim_time - last_shot_time >= 5:
                save_screenshot(screen, output_folder, f"02_progress_{int(sim_time)}s.png")
                last_shot_time = sim_time
            clock.tick(60) 

    if video_writer is not None:
        video_writer.release() 

    rock_displacements = [get_distance_pixels(b.x, b.y) - get_distance_pixels(b.start_x, b.start_y) for b in blocks]
    mean_rock_displacement = statistics.mean(rock_displacements)
    
    bbot_displacements = [get_distance_pixels(bbot.x, bbot.y) - get_distance_pixels(bbot.start_x, bbot.start_y) for bbot in bbots]
    mean_bbot_displacement = statistics.mean(bbot_displacements)

    stop_ticks = [bbot.stop_tick if bbot.stop_tick is not None else ticks for bbot in bbots]
    stop_times = [bbot.stop_time if bbot.stop_time is not None else final_time for bbot in bbots]

    if is_visual and screen:
        save_screenshot(screen, output_folder, "03_sim_finished.png")
        
        sd_rock_displacement = statistics.stdev(rock_displacements) if len(rock_displacements) > 1 else 0.0
        sd_bbot_displacement = statistics.stdev(bbot_displacements) if len(bbot_displacements) > 1 else 0.0
        
        m_stop_steps = statistics.mean(stop_ticks) if stop_ticks else 0.0
        sd_stop_steps = statistics.stdev(stop_ticks) if len(stop_ticks) > 1 else 0.0
        
        m_stop_time = statistics.mean(stop_times) if stop_times else 0.0
        sd_stop_time = statistics.stdev(stop_times) if len(stop_times) > 1 else 0.0
        
        ui_show_trial1_results(
            screen, final_time, ticks, 
            mean_rock_displacement, sd_rock_displacement, 
            mean_bbot_displacement, sd_bbot_displacement, 
            output_folder, 
            m_stop_time, sd_stop_time,
            m_stop_steps, sd_stop_steps, 
            target_mean, target_sd
        )

    return {
        'trial': trial_num, 
        'duration': final_time, 
        'duration_steps': ticks,    
        'mean_rock_displacement': mean_rock_displacement, 
        'rock_displacements': rock_displacements,
        'mean_bbot_displacement': mean_bbot_displacement,
        'bbot_displacements': bbot_displacements,
        'stop_ticks': stop_ticks,
        'stop_times': stop_times
    }

# =============================================================================
# SECTION 6: MASTER CONTROL
# =============================================================================
def main():
    """Main execution block."""
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DARK BBOT CONTROL")

    # Prompt user for inputs
    num_trials, target_mean, target_sd = ui_get_simulation_parameters(screen)

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = f"dark_bbot_output_{timestamp_str}"
    os.makedirs(output_folder, exist_ok=True)
    
    # Run Trial 1 visually
    all_results = [run_simulation(1, True, output_folder, target_mean, target_sd, screen=screen)]
    
    # Export dedicated Trial 1 stats to CSV
    export_trial_1_metrics(all_results[0], target_mean, target_sd, output_folder)
    
    # Run remaining trials heedlessly
    if num_trials > 1:
        for i in range(2, num_trials + 1):
            if i % 10 == 0 or i == 2: 
                ui_show_headless_progress(screen, i, num_trials)
            all_results.append(run_simulation(i, False, output_folder, target_mean, target_sd))

    # Compile data for final reporting (PURGING TRIAL 1)
    aggregate_results = [r for r in all_results if r['trial'] > 1]
    
    pooled_bbot_steps = []
    pooled_bbot_times = []
    pooled_rock_displacements = []
    pooled_bbot_displacements = []
    
    intra_run_rock_means = []
    intra_run_bbot_means = []
    intra_run_step_means = []
    intra_run_time_means = []
    
    for r in aggregate_results:
        st_steps = r['stop_ticks']
        st_time = r['stop_times']
        rd = r['rock_displacements']
        bd = r['bbot_displacements'] 
        
        pooled_bbot_steps.extend(st_steps)
        pooled_bbot_times.extend(st_time)
        pooled_rock_displacements.extend(rd)
        pooled_bbot_displacements.extend(bd) 
        
        intra_run_rock_means.append(r['mean_rock_displacement'])
        intra_run_bbot_means.append(r['mean_bbot_displacement'])
        intra_run_step_means.append(statistics.mean(st_steps) if st_steps else 0.0)
        intra_run_time_means.append(statistics.mean(st_time) if st_time else 0.0)
            
    final_stats = {
        'm_pool_rock': statistics.mean(pooled_rock_displacements) if pooled_rock_displacements else 0.0,
        'sd_pool_rock': statistics.stdev(pooled_rock_displacements) if len(pooled_rock_displacements) > 1 else 0.0,
        'm_intra_rock': statistics.mean(intra_run_rock_means) if intra_run_rock_means else 0.0,
        'sd_intra_rock': statistics.stdev(intra_run_rock_means) if len(intra_run_rock_means) > 1 else 0.0,
        
        'm_pool_bbot_disp': statistics.mean(pooled_bbot_displacements) if pooled_bbot_displacements else 0.0,
        'sd_pool_bbot_disp': statistics.stdev(pooled_bbot_displacements) if len(pooled_bbot_displacements) > 1 else 0.0,
        'm_intra_bbot_disp': statistics.mean(intra_run_bbot_means) if intra_run_bbot_means else 0.0,
        'sd_intra_bbot_disp': statistics.stdev(intra_run_bbot_means) if len(intra_run_bbot_means) > 1 else 0.0,
        
        'm_pool_steps': statistics.mean(pooled_bbot_steps) if pooled_bbot_steps else 0.0,
        'sd_pool_steps': statistics.stdev(pooled_bbot_steps) if len(pooled_bbot_steps) > 1 else 0.0,
        'm_intra_steps': statistics.mean(intra_run_step_means) if intra_run_step_means else 0.0,
        'sd_intra_steps': statistics.stdev(intra_run_step_means) if len(intra_run_step_means) > 1 else 0.0,
        
        'm_pool_time': statistics.mean(pooled_bbot_times) if pooled_bbot_times else 0.0,
        'sd_pool_time': statistics.stdev(pooled_bbot_times) if len(pooled_bbot_times) > 1 else 0.0,
        'm_intra_time': statistics.mean(intra_run_time_means) if intra_run_time_means else 0.0,
        'sd_intra_time': statistics.stdev(intra_run_time_means) if len(intra_run_time_means) > 1 else 0.0,
    }

    # Generate CSV and plots
    export_aggregate_summary(aggregate_results, output_folder, final_stats, target_mean, target_sd)
                             
    plot_histogram(pooled_bbot_times, 'Distribution of BBot Stop Times', 'Simulation Time (seconds)', 'Frequency', "05_bbot_times_hist.png", 'royalblue', output_folder, True)
    plot_histogram(pooled_rock_displacements, 'Distribution of Net Rock Displacements', 'Net Displacement (pixels)', 'Frequency', "06_rock_displacement_hist.png", 'forestgreen', output_folder)
    plot_histogram(pooled_bbot_displacements, 'Distribution of Net BBot Displacements', 'Net Displacement (pixels)', 'Frequency', "07_bbot_displacement_hist.png", 'darkorange', output_folder)
    
    ui_show_final_done(screen, output_folder)
    pygame.quit()

if __name__ == "__main__":
    main()