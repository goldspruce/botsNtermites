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
import glob         # Helps us easily find all .csv files in the directory.

# --- NEW IMPORTS FOR VIDEO EXPORT ---
import cv2          # OpenCV: The gold standard for video writing and image processing.
import numpy as np  # NumPy: Handles the heavy matrix math needed to convert screen pixels to a video file.

# =============================================================================
# SECTION 1: GLOBAL CONFIGURATION (CONSTANTS)
# =============================================================================

WIDTH, HEIGHT = 600, 600          # The dimensions of our simulation window in pixels.

BG_COLOR = (20, 20, 20)           # Dark Gray background (RGB color format).
DARK_MARKER_COLOR = (101, 67, 33) # Dark brown circle to denote the "dark" simulation.
LIGHT_POS = (15, 15)              # The (X, Y) coordinates of the dark marker.
LIGHT_RADIUS_VISUAL = 5           # Visual size of the marker.

BBOT_COLOR = (220, 60, 60)        # Red for bbots that are moving and searching.
REST_COLOR = (60, 220, 60)        # Green for bbots that have gone to sleep.
BLOCK_COLOR = (80, 80, 220)       # Blue for the pushable rocks.
WALL_COLOR = (255, 255, 0)        # Yellow for the boundaries.
TEXT_COLOR = (255, 255, 255)      # White text for UI elements.

# --- CORE SIMULATION NUMBERS ---
NUM_BBOTS = 50                    # We will spawn exactly 50 bbots every trial.
NUM_BLOCKS = 20                   # We will spawn exactly 20 rocks every trial.
BLOCK_RADIUS = 20                 # The radius for the circular rocks.
BBOT_RADIUS = 10                  # The radius of our circular bbots.
WALL_THICKNESS = 10               # How thick the yellow boundary lines are.

BBOT_SPEED_GO = 3.5               # How many pixels a bbot moves per frame while wandering normally.
BBOT_SPEED_PUSH = 1.0             # BBots slow down to this speed when they are actively touching a rock.
PUSH_FORCE = 1.0                  # How much momentum a bbot transfers to a rock when it rams it.
BLOCK_FRICTION = 0.90             # Every frame, a moving rock's speed is multiplied by 0.90 (it loses 10% of its speed).

SIM_TIMEOUT = 300.0               # Maximum seconds a trial can run before forcing a stop (preventing infinite loops).

# =============================================================================
# SECTION 2: HELPER FUNCTIONS & CLASSES
# =============================================================================

def save_screenshot(screen, folder, filename):
    """Takes a 'picture' of the Pygame window and saves it as a PNG image."""
    path = os.path.join(folder, filename) 
    pygame.image.save(screen, path)

def get_bbot_spawn_pos(existing_bbots):
    """Finds a random starting coordinate for a bbot, ensuring it doesn't overlap other bbots."""
    safe_min = WALL_THICKNESS + BBOT_RADIUS
    safe_max_x = WIDTH - WALL_THICKNESS - BBOT_RADIUS
    safe_max_y = HEIGHT - WALL_THICKNESS - BBOT_RADIUS
    
    while True:
        x = random.uniform(safe_min, safe_max_x) 
        y = random.uniform(safe_min, safe_max_y) 
        
        # Check against already spawned bbots
        if any(math.hypot(x - b.x, y - b.y) < (BBOT_RADIUS * 2 + 2) for b in existing_bbots):
            continue
            
        return x, y 

def get_valid_block_pos(existing_bbots, existing_blocks):
    """Finds a random starting coordinate for a rock. Ensures it doesn't spawn on a bbot OR another rock."""
    safe_min = WALL_THICKNESS + BLOCK_RADIUS
    safe_max_x = WIDTH - WALL_THICKNESS - BLOCK_RADIUS
    safe_max_y = HEIGHT - WALL_THICKNESS - BLOCK_RADIUS
    
    while True: 
        x = random.randint(int(safe_min), int(safe_max_x))
        y = random.randint(int(safe_min), int(safe_max_y))
        
        # Check against all existing bbots
        if any(math.hypot(x - b.x, y - b.y) < (BLOCK_RADIUS + BBOT_RADIUS + 2) for b in existing_bbots):
            continue 
            
        # Check against all previously spawned rocks
        if any(math.hypot(x - blk.x, y - blk.y) < (BLOCK_RADIUS * 2 + 2) for blk in existing_blocks):
            continue
            
        return x, y

def get_distance_pixels(x, y):
    """Calculates how far an (X,Y) point is from the absolute center of the room."""
    center_x = WIDTH / 2
    center_y = HEIGHT / 2
    return math.hypot(x - center_x, y - center_y)
    	
def draw_text(screen, text, size, x, y, color=TEXT_COLOR):
    """Helper to cleanly draw words on the Pygame window."""
    font = pygame.font.SysFont("Arial", size, bold=False)
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y)) 

# --- CLASSES ---

class BBot:
    """The blueprint for our Braitenberg Vehicles."""
    def __init__(self, target_mean, target_sd, existing_bbots):
        # Starting positions and randomized initial direction
        self.x, self.y = get_bbot_spawn_pos(existing_bbots) 
        self.start_x, self.start_y = self.x, self.y
        self.angle = random.uniform(0, 2 * math.pi) 
        
        self.touching_block = False 
        self.is_stopped = False
        self.stop_tick = None
        self.stop_time = None 
        
        # STOCHASTIC STOPPAGE TIMER
        # Assign a random target stop step based on the loaded (or default) stats.
        while True:
            t = random.gauss(target_mean, target_sd)
            if t > 0: 
                self.target_stop_steps = t
                break

    def decide_and_move(self, current_steps):
        """Controls the movement behavior of the BBot based on its stochastic timer."""
        
        # 1. ELASTIC WALL BOUNCE
        # We put this at the top so sleeping bots cleanly deflect if pushed into a wall.
        min_bound = WALL_THICKNESS + BBOT_RADIUS
        max_bound = WIDTH - min_bound
        
        # Check X boundaries (Left/Right Walls)
        if self.x <= min_bound:
            self.x = min_bound
            # Reflect angle across the Y-axis
            self.angle = math.pi - self.angle 
        elif self.x >= max_bound:
            self.x = max_bound
            # Reflect angle across the Y-axis
            self.angle = math.pi - self.angle

        # Check Y boundaries (Top/Bottom Walls)
        max_y_bound = HEIGHT - min_bound
        if self.y <= min_bound:
            self.y = min_bound
            # Reflect angle across the X-axis
            self.angle = -self.angle
        elif self.y >= max_y_bound:
            self.y = max_y_bound
            # Reflect angle across the X-axis
            self.angle = -self.angle
            
        # Ensure angle stays within 0 to 2*PI for clean math
        self.angle = self.angle % (2 * math.pi)

        # 2. Check the stochastic timer
        if current_steps >= self.target_stop_steps:
            self.is_stopped = True

        # 3. Exit if stopped (Now safe to do, because walls were already checked!)
        if self.is_stopped: return 
        
        # 4. Movement Logic
        # Drop to pushing speed if struggling against a rock, otherwise go normal speed.
        speed = BBOT_SPEED_PUSH if self.touching_block else BBOT_SPEED_GO
        
        # THE JITTER MECHANIC:
        # BRAIN-DEAD MODE: The BBot constantly twitches its steering wheel up to ~23 degrees,
        # even when touching a rock. This removes its ability to reliably bulldoze blocks.
        self.angle += random.uniform(-0.4, 0.4)

        # Apply velocity based on current angle
        self.x += math.cos(self.angle) * speed
        self.y += math.sin(self.angle) * speed

class Block:
    """The blueprint for our pushable rocks."""
    def __init__(self, existing_bbots, existing_blocks):
        self.radius = BLOCK_RADIUS
        self.x, self.y = get_valid_block_pos(existing_bbots, existing_blocks)
        self.start_x, self.start_y = self.x, self.y 
        self.vx, self.vy = 0.0, 0.0 

    def update(self):
        """Updates the rock's physical position every frame."""
        self.x += self.vx
        self.y += self.vy
        
        # Exponential decay of momentum (Friction)
        self.vx *= BLOCK_FRICTION
        self.vy *= BLOCK_FRICTION
        
        # ELASTIC WALL BOUNCE FOR ROCKS
        min_bound = WALL_THICKNESS + self.radius
        max_x_bound = WIDTH - min_bound
        max_y_bound = HEIGHT - min_bound

        # Check X boundaries (Left/Right Walls)
        if self.x <= min_bound: 
            self.x = min_bound
            self.vx = -self.vx  # Reverse X momentum
        elif self.x >= max_x_bound: 
            self.x = max_x_bound
            self.vx = -self.vx  # Reverse X momentum
            
        # Check Y boundaries (Top/Bottom Walls)
        if self.y <= min_bound: 
            self.y = min_bound
            self.vy = -self.vy  # Reverse Y momentum
        elif self.y >= max_y_bound: 
            self.y = max_y_bound
            self.vy = -self.vy  # Reverse Y momentum

def check_collisions(bbots, blocks):
    """The Physics Engine: Handles all the bumping and pushing mathematically."""
    for bbot in bbots: 
        bbot.touching_block = False 
        for block in blocks:
            dx = block.x - bbot.x
            dy = block.y - bbot.y
            dist = math.hypot(dx, dy)
            
            # BBot colliding with Rock
            if dist < (BBOT_RADIUS + block.radius):
                bbot.touching_block = True
                if dist < 0.1: dist = 0.1 
                nx, ny = dx / dist, dy / dist
                block.vx += nx * PUSH_FORCE
                block.vy += ny * PUSH_FORCE
                bbot.x -= nx * 2
                bbot.y -= ny * 2

    # Rock colliding with Rock
    for i, b1 in enumerate(blocks):
        for b2 in blocks[i+1:]: 
            dx = b1.x - b2.x
            dy = b1.y - b2.y
            dist = math.hypot(dx, dy)
            if dist < (b1.radius + b2.radius):
                if dist < 0.1: dist = 0.1
                nx, ny = dx / dist, dy / dist
                b1.x += nx * 2.0; b1.y += ny * 2.0
                b2.x -= nx * 2.0; b2.y -= ny * 2.0

# =============================================================================
# SECTION 3: EXPORTING, REPORTING & PLOTTING
# =============================================================================

def extract_stats_from_csv(filepath):
    """Reads a CSV file to extract Total Trials, Mean Stop Steps, and SD Stop Steps."""
    num_trials = 10
    target_mean = 815.72
    target_sd = 704.3755
    
    try:
        with open(filepath, mode='r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(',')]
                if not parts: continue
                
                if "Total Trials" in parts[0] or "Total Aggregated Trials" in parts[0]:
                    if len(parts) > 1 and parts[1]:
                        num_trials = int(parts[1])
                
                if len(parts) >= 3 and "Mean of Intra-Run Stop Time Means (steps)" in parts[1]:
                    target_mean = float(parts[2])
                    if i + 1 < len(lines):
                        next_parts = [p.strip() for p in lines[i+1].split(',')]
                        if len(next_parts) >= 3 and "SD of Above (steps)" in next_parts[1]:
                            target_sd = float(next_parts[2])
                            
    except Exception as e:
        print(f"Error reading CSV {filepath}: {e}")
        
    return num_trials, target_mean, target_sd

def export_trial_01_metrics(result, output_folder, target_mean, target_sd):
    """Saves a dedicated CSV for just Trial 1's results without rounding."""
    filename = os.path.join(output_folder, "trial_01_metrics.csv")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["trial_01_metrics.csv"])
        writer.writerow([])
        
        writer.writerow(["Target Mean Stop Time (steps)", target_mean])
        writer.writerow(["Target SD of Stop Times (steps)", target_sd])
        writer.writerow([])
        
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
        
        rd = result['rock_displacements']
        sd_rock = statistics.stdev(rd) if len(rd) > 1 else 0.0
        writer.writerow(["Mean Rock Displacement (px)", result['mean_rock_displacement']])
        writer.writerow(["Rock Displacement SD (px)", sd_rock])
        writer.writerow([])
        
        bd = result['bbot_displacements']
        sd_bbot = statistics.stdev(bd) if len(bd) > 1 else 0.0
        writer.writerow(["Mean BBot Displacement (px)", result['mean_bbot_displacement']])
        writer.writerow(["BBot Displacement SD (px)", sd_bbot])

def export_aggregate_summary(all_results, output_folder, final_stats, target_mean, target_sd):
    """Saves the master summary incorporating intra-run and pooled statistics. Trial 1 is guaranteed to be here."""
    filename = os.path.join(output_folder, "dark_aggregate_summary.csv")

    bad_rock_trials = [str(r['trial']) for r in all_results if r['mean_rock_displacement'] <= 0]
    bad_bbot_trials = [str(r['trial']) for r in all_results if r['mean_bbot_displacement'] <= 0]

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(["aggregate_summary.csv"])
        writer.writerow([])
        
        writer.writerow(["Target Mean Stop Time (steps)", target_mean])
        writer.writerow(["Target SD of Stop Times (steps)", target_sd])
        writer.writerow([])
        
        writer.writerow(["Trials with Mean Rock Displacement <= 0", len(bad_rock_trials)])
        writer.writerow(["Specific Rock Trial IDs", ", ".join(bad_rock_trials) if bad_rock_trials else "None"])
        writer.writerow([])
        writer.writerow(["Trials with Mean BBot Displacement <= 0", len(bad_bbot_trials)])
        writer.writerow(["Specific BBot Trial IDs", ", ".join(bad_bbot_trials) if bad_bbot_trials else "None"])
        writer.writerow([])
        
        writer.writerow(["Total Trials", len(all_results)])
        writer.writerow([])
        
        writer.writerow(["", "Mean of Pooled Rock Displacement (px)", final_stats['m_pool_rock']])
        writer.writerow(["", "SD of Above (px)", final_stats['sd_pool_rock']])
        writer.writerow(["", "Mean of Intra-Run Rock Displacement (px)", final_stats['m_intra_rock']])
        writer.writerow(["", "SD of Above (px)", final_stats['sd_intra_rock']])
        writer.writerow([])
        
        writer.writerow(["", "Mean of Pooled Bbot Displacements (px)", final_stats['m_pool_bbot_disp']])
        writer.writerow(["", "SD of Above (px)", final_stats['sd_pool_bbot_disp']])
        writer.writerow(["", "Mean of Intra-Run Bbot Displacements (px)", final_stats['m_intra_bbot_disp']])
        writer.writerow(["", "SD of Above (px)", final_stats['sd_intra_bbot_disp']])
        writer.writerow([])
        
        writer.writerow(["", "Mean of Pooled Bbot Stop Times (steps)", final_stats['m_pool_steps']])
        writer.writerow(["", "SD of Above (steps)", final_stats['sd_pool_steps']])
        writer.writerow(["", "Mean of Intra-Run Stop Time Means (steps)", final_stats['m_intra_steps']])
        writer.writerow(["", "SD of Above (steps)", final_stats['sd_intra_steps']])
        writer.writerow([])
        
        writer.writerow(["", "Mean of Pooled Bbot Stop Times (s)", final_stats['m_pool_time']])
        writer.writerow(["", "SD of Above (s)", final_stats['sd_pool_time']])
        writer.writerow(["", "Mean of Intra-Run Stop Time Means (s)", final_stats['m_intra_time']])
        writer.writerow(["", "SD of Above (s)", final_stats['sd_intra_time']])
        writer.writerow([])
        writer.writerow([])
        
        # Trial By Trial Breakdown (Trial 1 is explicitly written here along with the rest)
        writer.writerow([
            "Trial", "Duration (steps)", "Duration (s)", 
            "Mean Rock Displacement", "Rock Displacement SD", 
            "Mean BBot Displacement", "BBot Displacement SD", 
            "Mean BBot Stop Time (steps)", "BBot Stop SD (steps)",
            "Mean BBot Stop Time (s)", "BBot Stop SD (s)"
        ])
        for r in all_results:
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
    """Generates and saves visual charts (histograms) of our data distributions."""
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

def ui_select_csv_file(screen):
    """Interactive File Explorer inside Pygame to navigate folders and pick a CSV."""
    current_dir = os.getcwd()
    
    selected_idx = 0
    scroll_idx = 0
    max_display = 12 # Max items on screen before scrolling
    clock = pygame.time.Clock()
    
    while True:
        # 1. Fetch current directory contents
        try:
            all_items = os.listdir(current_dir)
        except PermissionError:
            all_items = []
            
        # 2. Sort into Directories and CSVs
        folders = [f for f in all_items if os.path.isdir(os.path.join(current_dir, f))]
        folders.sort()
        
        csv_files = [f for f in all_items if f.endswith('.csv') and os.path.isfile(os.path.join(current_dir, f))]
        csv_files.sort()
        
        # 3. Build the UI list
        options = ["[..] GO UP A FOLDER"]
        options.extend([f"[DIR] {folder}" for folder in folders])
        options.extend(csv_files)
        options.append("-- Use Default Settings (10 Trials, M: 815.72, SD: 704.38) --")
        
        # Prevent index out of bounds if directory changes
        if selected_idx >= len(options): selected_idx = len(options) - 1
        
        # --- DRAWING ---
        screen.fill(BG_COLOR)
        draw_text(screen, "SELECT PREVIOUS SIMULATION CSV TO LOAD STATS", 24, WIDTH//2 - 250, 20, (255, 215, 0))
        
        # Show Current Path
        path_display = current_dir if len(current_dir) < 60 else "..." + current_dir[-57:]
        draw_text(screen, f"Path: {path_display}", 16, 20, 60, (200, 200, 200))
        draw_text(screen, "(Use UP/DOWN arrows and press ENTER to select)", 16, WIDTH//2 - 180, 85, (200, 200, 200))
        
        # Paginated Drawing
        for i in range(scroll_idx, min(len(options), scroll_idx + max_display)):
            text = options[i]
            
            # Determine Color
            if i == selected_idx:
                color = (100, 255, 100) # Green highlight
            elif text == "[..] GO UP A FOLDER":
                color = (255, 100, 100) # Red for back
            elif text.startswith("[DIR]"):
                color = (100, 200, 255) # Light blue for folders
            elif text == options[-1]:
                color = (200, 150, 50)  # Orange for fallback
            else:
                color = (255, 255, 255) # White for CSVs
                
            prefix = "-> " if i == selected_idx else "   "
            display_y = 120 + (i - scroll_idx) * 35
            draw_text(screen, prefix + text, 20, 40, display_y, color)
            
        pygame.display.flip()
        
        # --- EVENT HANDLING ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_idx = max(0, selected_idx - 1)
                    if selected_idx < scroll_idx:
                        scroll_idx = selected_idx
                elif event.key == pygame.K_DOWN:
                    selected_idx = min(len(options) - 1, selected_idx + 1)
                    if selected_idx >= scroll_idx + max_display:
                        scroll_idx += 1
                        
                elif event.key == pygame.K_RETURN:
                    selected_text = options[selected_idx]
                    
                    if selected_text == "[..] GO UP A FOLDER":
                        current_dir = os.path.dirname(current_dir)
                        selected_idx, scroll_idx = 0, 0
                        
                    elif selected_text.startswith("[DIR] "):
                        folder_name = selected_text.replace("[DIR] ", "")
                        current_dir = os.path.join(current_dir, folder_name)
                        selected_idx, scroll_idx = 0, 0
                        
                    elif selected_text == options[-1]:
                        # Default fallback chosen
                        return None
                        
                    else:
                        # A .csv file was selected!
                        return os.path.join(current_dir, selected_text)
                        
        clock.tick(30)

def ui_show_headless_progress(screen, current, total):
    """Displays a loading screen while background simulations are crunching."""
    screen.fill(BG_COLOR)
    draw_text(screen, "RUNNING HEADLESS SIMULATIONS", 28, WIDTH//2 - 200, HEIGHT//2 - 40, (255, 215, 0))
    draw_text(screen, f"Crunching Trial {current} / {total}...", 20, WIDTH//2 - 100, HEIGHT//2 + 10)
    pygame.display.flip()
    pygame.event.pump() 

def ui_show_trial1_results(screen, duration_secs, duration_steps, mean_rock_displacement, sd_rock_displacement, mean_bbot_displacement, sd_bbot_displacement, output_folder, mean_stop_time, sd_stop_time, mean_stop_steps, sd_stop_steps, target_mean, target_sd, total_trials):
    """Pauses the program after Trial 1 to show the user the baseline stats vs the targets."""
    waiting = True
    screenshot_taken = False
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN: waiting = False

        screen.fill(BG_COLOR)
        draw_text(screen, "TRIAL 1 COMPLETE", 32, 30, 20, (255, 215, 0))
        
        # Duration
        draw_text(screen, f"Duration (Steps):      {duration_steps}", 18, 30, 70)
        draw_text(screen, f"Duration (Video Secs): {duration_secs:.2f} s", 18, 30, 95)
        
        # Target vs Actual Stats
        draw_text(screen, "--- TARGETS (Loaded/Hardcoded) ---", 20, 30, 130, (100, 255, 100))
        draw_text(screen, f"Total Trials: {total_trials}", 18, 30, 155)
        draw_text(screen, f"Mean Stop:    {target_mean:.2f} steps", 18, 30, 180)
        draw_text(screen, f"SD of Stops:  {target_sd:.2f} steps", 18, 30, 205)
        
        # THE BLUE TITLE (Y = 245)
        draw_text(screen, "--- ACTUAL TRIAL 1 RESULTS ---", 20, 30, 245, (100, 255, 255))
        
        # Left Column: Stoppage Stats
        draw_text(screen, f"Mean Stop:    {mean_stop_steps:.2f} steps", 18, 30, 275)
        draw_text(screen, f"SD of Stops:  {sd_stop_steps:.2f} steps", 18, 30, 300)
        
        # Right Column: Displacement Stats (All moved BELOW Y=245)
        sign_rock = "" if mean_rock_displacement < 0 else "+"
        draw_text(screen, f"Mean Rock Disp:      {sign_rock}{mean_rock_displacement:.2f} px", 18, 330, 275)
        draw_text(screen, f"Rock Disp SD:        {sd_rock_displacement:.2f} px", 18, 330, 300, (200, 200, 200))
        
        sign_bbot = "" if mean_bbot_displacement < 0 else "+"
        draw_text(screen, f"Mean BBot Disp:      {sign_bbot}{mean_bbot_displacement:.2f} px", 18, 330, 335)
        draw_text(screen, f"BBot Disp SD:        {sd_bbot_displacement:.2f} px", 18, 330, 360, (200, 200, 200))

        draw_text(screen, "Press [ENTER] to begin Headless Runs.", 22, 30, HEIGHT - 50, (255, 100, 100))
        pygame.display.flip()
        
        if not screenshot_taken:
            save_screenshot(screen, output_folder, "04_final_metrics.png")
            screenshot_taken = True 
        pygame.time.Clock().tick(15)

def ui_show_final_done(screen, output_folder):
    """The final screen telling the user where to find their CSVs and Histograms."""
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

def run_simulation(trial_num, is_visual, output_folder, target_mean, target_sd, total_trials, screen=None):
    """Executes a single trial run of the physics simulation."""
    
    # Spawn BBots iteratively so they can check against each other
    bbots = []
    for _ in range(NUM_BBOTS):
        bbots.append(BBot(target_mean, target_sd, bbots))
        
    # Spawn Rocks iteratively so they can check against bbots AND previous rocks
    blocks = []
    for _ in range(NUM_BLOCKS):
        blocks.append(Block(bbots, blocks))
        
    ticks, sim_time, final_time = 0, 0.0, None
    video_writer = None 
        
    # --- VIDEO & VISUAL SETUP FOR TRIAL 1 ---
    if is_visual and screen:
        pygame.display.set_caption("DARK BBOT CONTROL")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_filename = os.path.join(output_folder, "00_trial_01_video.mp4")
        video_writer = cv2.VideoWriter(video_filename, fourcc, 60.0, (WIDTH, HEIGHT))
        
        clock = pygame.time.Clock()
        
        # We wait for the user to explicitly start the simulation
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
            
            # The brown circle to explicitly denote the Dark Environment
            pygame.draw.circle(screen, DARK_MARKER_COLOR, LIGHT_POS, LIGHT_RADIUS_VISUAL)
            
            for b in blocks: pygame.draw.circle(screen, BLOCK_COLOR, (int(b.x), int(b.y)), int(b.radius))
            for bbot in bbots: pygame.draw.circle(screen, BBOT_COLOR, (int(bbot.x), int(bbot.y)), BBOT_RADIUS)
            
            # Exactly matching requested output format on the pretrial screen
            draw_text(screen, f"Baselines Trials: {total_trials} - Mean:{target_mean:.2f} - SD: {target_sd:.2f}", 20, WIDTH//2 - 200, 60, (100, 255, 100))
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
            # Send step count so bbot checks its randomized stochastic timer
            bbot.decide_and_move(ticks) 
            
            if bbot.is_stopped:
                safe_bbots += 1
                if bbot.stop_tick is None:
                    bbot.stop_tick = ticks
                    bbot.stop_time = sim_time 
                
        # End conditions: All bbots asleep, or timeout reached
        if safe_bbots == NUM_BBOTS or sim_time >= SIM_TIMEOUT:
            final_time = sim_time
            running = False 
            
        # Draw graphics if running in Visual Mode
        if is_visual and screen:
            screen.fill(BG_COLOR)
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, HEIGHT-WALL_THICKNESS, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WALL_THICKNESS, HEIGHT))
            pygame.draw.rect(screen, WALL_COLOR, (WIDTH-WALL_THICKNESS, 0, WALL_THICKNESS, HEIGHT))
            
            # The brown circle to explicitly denote the Dark Environment
            pygame.draw.circle(screen, DARK_MARKER_COLOR, LIGHT_POS, LIGHT_RADIUS_VISUAL)
            
            # Explicitly render both simulated time and exact step count
            draw_text(screen, f"{sim_time:.2f}s | {ticks} steps", 20, 10, 80, (255, 255, 0))
            
            for b in blocks: pygame.draw.circle(screen, BLOCK_COLOR, (int(b.x), int(b.y)), int(b.radius))
            for bbot in bbots:
                color = REST_COLOR if bbot.is_stopped else BBOT_COLOR
                pygame.draw.circle(screen, color, (int(bbot.x), int(bbot.y)), BBOT_RADIUS)
                
            pygame.display.flip() 
            
            # --- WRITE VIDEO FRAME ---
            if video_writer is not None:
                frame = pygame.surfarray.array3d(screen)
                frame = frame.transpose([1, 0, 2])
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)
            
            # Take a screenshot every 5 seconds
            if sim_time - last_shot_time >= 5:
                save_screenshot(screen, output_folder, f"02_progress_{int(sim_time)}s.png")
                last_shot_time = sim_time
            clock.tick(60) 

    # --- TRIAL CLEANUP ---
    if video_writer is not None:
        video_writer.release() 

    # Collect final math for this specific run
    rock_displacements = [get_distance_pixels(b.x, b.y) - get_distance_pixels(b.start_x, b.start_y) for b in blocks]
    mean_rock_displacement = statistics.mean(rock_displacements)
    
    bbot_displacements = [get_distance_pixels(bbot.x, bbot.y) - get_distance_pixels(bbot.start_x, bbot.start_y) for bbot in bbots]
    mean_bbot_displacement = statistics.mean(bbot_displacements)

    stop_ticks = [bbot.stop_tick if bbot.stop_tick is not None else ticks for bbot in bbots]
    stop_times = [bbot.stop_time if bbot.stop_time is not None else final_time for bbot in bbots]

    # Process final Trial 1 outputs and PAUSE BEFORE UI STATS
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
            output_folder, m_stop_time, sd_stop_time,
            m_stop_steps, sd_stop_steps,
            target_mean, target_sd, total_trials
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
    """The central brain that organizes the folders, loops the trials, and processes the final stats."""
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DARK BBOT CONTROL")

    # Flexible CSV Menu Execution (NOW SUPPORTS FOLDERS!)
    csv_filename = ui_select_csv_file(screen)
    
    if csv_filename is None:
        # User selected the "Default Settings" fallback option
        num_trials, target_mean, target_sd = 10, 815.72, 704.3755
    else:
        num_trials, target_mean, target_sd = extract_stats_from_csv(csv_filename)

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = f"dark_bbot_output_{timestamp_str}"
    os.makedirs(output_folder, exist_ok=True)
    
    # Always run Trial 1 visually
    all_results = [run_simulation(1, True, output_folder, target_mean, target_sd, num_trials, screen=screen)]
    
    # Export dedicated Trial 1 stats
    export_trial_01_metrics(all_results[0], output_folder, target_mean, target_sd)
    
    # Run remaining trials headlessly
    if num_trials > 1:
        for i in range(2, num_trials + 1):
            if i % 10 == 0 or i == 2: 
                ui_show_headless_progress(screen, i, num_trials)
            all_results.append(run_simulation(i, False, output_folder, target_mean, target_sd, num_trials))

    # --- POOLING AND ANALYZING INTRA-RUN MEANS (USING ALL TRIALS INCLUDING TRIAL 1) ---
    pooled_bbot_steps = []
    pooled_bbot_times = []
    pooled_rock_displacements = []
    pooled_bbot_displacements = []
    
    intra_run_rock_means = []
    intra_run_bbot_means = []
    intra_run_step_means = []
    intra_run_time_means = []
    
    for r in all_results:
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
            
    # --- CALCULATING THE REQUESTED METRICS ---
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

    # --- EXPORT AND PLOT ---
    export_aggregate_summary(all_results, output_folder, final_stats, target_mean, target_sd)
                             
    plot_histogram(pooled_bbot_times, 'Distribution of BBot Stop Times', 'Simulation Time (seconds)', 'Frequency', "05_bbot_times_hist.png", 'royalblue', output_folder, True)
    plot_histogram(pooled_rock_displacements, 'Distribution of Net Rock Displacements', 'Net Displacement (pixels)', 'Frequency', "06_rock_displacement_hist.png", 'forestgreen', output_folder)
    plot_histogram(pooled_bbot_displacements, 'Distribution of Net BBot Displacements', 'Net Displacement (pixels)', 'Frequency', "07_bbot_displacement_hist.png", 'darkorange', output_folder)
    
    ui_show_final_done(screen, output_folder)
    pygame.quit()

if __name__ == "__main__":
    main()