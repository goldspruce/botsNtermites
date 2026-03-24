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

WIDTH, HEIGHT = 600, 600          # The dimensions of our simulation window in pixels.

BG_COLOR = (20, 20, 20)           # Dark Gray background (RGB color format).
LIGHT_COLOR = (255, 255, 240)     # Off-White for the light source.
LIGHT_POS = (15, 15)              # The (X, Y) coordinates of the light source in the top-left corner.
LIGHT_RADIUS_VISUAL = 5           # How big the light source looks on the screen.

BBOT_COLOR = (220, 60, 60)        # Red for bbots that are moving and searching.
REST_COLOR = (60, 220, 60)        # Green for bbots that have found a shadow and gone to sleep.
BLOCK_COLOR = (80, 80, 220)       # Blue for the pushable rocks.
WALL_COLOR = (255, 255, 0)        # Yellow for the boundaries.
TEXT_COLOR = (255, 255, 255)      # White text for UI elements.

# --- CORE SIMULATION NUMBERS ---
NUM_BBOTS = 50                    # We will spawn exactly 50 bbots every trial.
NUM_BLOCKS = 20                   # We will spawn exactly 20 rocks every trial.
BLOCK_RADIUS = 20                 # The radius for the circular rocks.
BBOT_RADIUS = 10                  # The radius of our circular bbots.
WALL_THICKNESS = 10               # How thick the yellow boundary lines are.

SHADOW_PROXIMITY_LIMIT = 40       # The max distance a bbot can be from a rock to consider itself "in a shadow".
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

def get_bbot_spawn_pos():
    """Finds a random starting (X, Y) coordinate for a bbot anywhere inside the walls."""
    safe_min = WALL_THICKNESS + BBOT_RADIUS
    safe_max_x = WIDTH - WALL_THICKNESS - BBOT_RADIUS
    safe_max_y = HEIGHT - WALL_THICKNESS - BBOT_RADIUS
    
    x = random.uniform(safe_min, safe_max_x) 
    y = random.uniform(safe_min, safe_max_y) 
    return x, y 

def get_valid_block_pos(existing_bbots):
    """Finds a random starting coordinate for a rock. Ensures it doesn't spawn directly on top of a bbot."""
    safe_min = WALL_THICKNESS + BLOCK_RADIUS
    safe_max_x = WIDTH - WALL_THICKNESS - BLOCK_RADIUS
    safe_max_y = HEIGHT - WALL_THICKNESS - BLOCK_RADIUS
    
    while True: 
        x = random.randint(int(safe_min), int(safe_max_x))
        y = random.randint(int(safe_min), int(safe_max_y))
        
        # Check distance against all existing bbots. If it's too close, loop again and pick a new spot.
        if any(math.hypot(x - b.x, y - b.y) < (BLOCK_RADIUS + BBOT_RADIUS + 2) for b in existing_bbots):
            continue 
            
        return x, y 

def get_distance_pixels(x, y):
    """A tiny shortcut function using Pythagorean theorem to find how far an (X,Y) point is from the light source."""
    return math.hypot(x - LIGHT_POS[0], y - LIGHT_POS[1])

def draw_text(screen, text, size, x, y, color=TEXT_COLOR):
    """Helper to cleanly draw words on the Pygame window."""
    font = pygame.font.SysFont("Arial", size, bold=False)
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y)) 

# --- CLASSES ---

class BBot:
    """The blueprint for our Braitenberg Vehicles."""
    def __init__(self):
        # Starting positions and randomized initial direction
        self.x, self.y = get_bbot_spawn_pos() 
        self.start_x, self.start_y = self.x, self.y
        self.angle = random.uniform(0, 2 * math.pi) 
        
        # Internal state tracking
        self.touching_block = False 
        self.in_shadow = False
        self.is_stopped = False
        self.stop_tick = None
        self.stop_time = None 

    def sense(self, blocks, light_pos):
        """Checks if the BBot is currently hidden from the light by a rock."""
        self.in_shadow = False
        self.touching_block = False
        
        # Calculate angle to the light source
        dx = self.x - light_pos[0]
        dy = self.y - light_pos[1]
        dist_to_light = math.hypot(dx, dy)
        
        if dist_to_light == 0: return 

        # Cast a virtual "ray" from the light source to the bbot
        dir_x = -(dx / dist_to_light)
        dir_y = -(dy / dist_to_light)
        
        sensor_x = self.x + (dir_x * BBOT_RADIUS)
        sensor_y = self.y + (dir_y * BBOT_RADIUS)
        
        line_start = light_pos
        line_end = (sensor_x, sensor_y)
        
        # Check if that ray hits any rocks before it hits the bbot
        for block in blocks:
            bx, by = block.x - block.radius, block.y - block.radius
            rect = pygame.Rect(int(bx), int(by), int(block.radius * 2), int(block.radius * 2))
            
            # If the light ray hits the rock, AND the bbot is close enough to the rock, it's in a shadow!
            if rect.clipline(line_start, line_end):
                dist_bbot_block = math.hypot(self.x - block.x, self.y - block.y)
                if dist_bbot_block <= (SHADOW_PROXIMITY_LIMIT + BBOT_RADIUS + block.radius):
                    self.in_shadow = True 
                    break 

    def decide_and_move(self):
        """Controls the movement behavior of the BBot based on its environment."""
        
        # 1. ELASTIC WALL BOUNCE
        min_bound = WALL_THICKNESS + BBOT_RADIUS
        max_bound = WIDTH - min_bound
        
        # Check X boundaries
        if self.x <= min_bound:
            self.x = min_bound
            # Reflect angle across the Y-axis
            self.angle = math.pi - self.angle 
        elif self.x >= max_bound:
            self.x = max_bound
            # Reflect angle across the Y-axis
            self.angle = math.pi - self.angle

        # Check Y boundaries
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
        
        # If safely in a shadow, cut the motor entirely.
        if self.is_stopped: return 
        
        # Drop to pushing speed if struggling against a rock, otherwise go normal speed.
        speed = BBOT_SPEED_PUSH if self.touching_block else BBOT_SPEED_GO
        
        # THE JITTER MECHANIC:
        # If NOT touching a block, randomly twitch the steering wheel up to ~23 degrees.
        # This creates a chaotic "Random Walk" ensuring the bbot eventually explores the whole arena.
        # By turning this OFF when touching a rock, the bbot pushes in a solid straight line.
        #if not self.touching_block: 
        #    self.angle += random.uniform(-0.4, 0.4)
        # THE JITTER MECHANIC:
        # BRAIN-DEAD MODE: The BBot constantly twitches its steering wheel up to ~23 degrees,
        # even when touching a rock. This removes its ability to reliably bulldoze blocks.
        self.angle += random.uniform(-0.4, 0.4)

        # Apply velocity based on current angle
        self.x += math.cos(self.angle) * speed
        self.y += math.sin(self.angle) * speed

class Block:
    """The blueprint for our pushable rocks."""
    def __init__(self, existing_bbots):
        self.radius = BLOCK_RADIUS
        self.x, self.y = get_valid_block_pos(existing_bbots)
        self.start_x, self.start_y = self.x, self.y 
        self.vx, self.vy = 0.0, 0.0 # Starts with zero velocity (standing still)

    def update(self):
        """Updates the rock's physical position every frame."""
        self.x += self.vx
        self.y += self.vy
        
        # THE FRICTION MECHANIC:
        # Multiply current speed by 0.90. This creates an exponential decay.
        # Once bbots stop pushing, the rock rapidly grinds to a halt.
        self.vx *= BLOCK_FRICTION
        self.vy *= BLOCK_FRICTION
        
        # Keep the rock trapped inside the yellow walls
        min_bound = WALL_THICKNESS + self.radius
        if self.x < min_bound: self.x = min_bound; self.vx = 0
        if self.x > WIDTH - min_bound: self.x = WIDTH - min_bound; self.vx = 0
        if self.y < min_bound: self.y = min_bound; self.vy = 0
        if self.y > HEIGHT - min_bound: self.y = HEIGHT - min_bound; self.vy = 0

def check_collisions(bbots, blocks):
    """The Physics Engine: Handles all the bumping and pushing mathematically."""
    
    # 1. BBot vs Rock Collisions (Circle vs. Circle)
    for bbot in bbots: 
        bbot.touching_block = False # Reset flag each frame
        for block in blocks:
            # Measure distance from center of bbot to center of rock
            dx = block.x - bbot.x
            dy = block.y - bbot.y
            dist = math.hypot(dx, dy)
            
            # If distance is less than their combined radii, they are overlapping!
            if dist < (BBOT_RADIUS + block.radius):
                bbot.touching_block = True
                if dist < 0.1: dist = 0.1 # Prevent dividing by zero in rare instances
                
                # Normalize the vector (create a unit arrow pointing from bbot to rock)
                nx, ny = dx / dist, dy / dist
                
                # Transfer momentum to the rock (The Push)
                block.vx += nx * PUSH_FORCE
                block.vy += ny * PUSH_FORCE
                
                # Recoil the bbot slightly backwards to prevent it from clipping inside the rock
                bbot.x -= nx * 2
                bbot.y -= ny * 2

    # 2. Rock vs Rock Collisions (Circle vs. Circle)
    for i, b1 in enumerate(blocks):
        for b2 in blocks[i+1:]: 
            dx = b1.x - b2.x
            dy = b1.y - b2.y
            dist = math.hypot(dx, dy)
            
            # If two rocks hit each other, bounce them apart evenly.
            if dist < (b1.radius + b2.radius):
                if dist < 0.1: dist = 0.1
                nx, ny = dx / dist, dy / dist
                b1.x += nx * 2.0; b1.y += ny * 2.0
                b2.x -= nx * 2.0; b2.y -= ny * 2.0

# =============================================================================
# SECTION 3: EXPORTING, REPORTING & PLOTTING
# =============================================================================

def export_trial_01_metrics(result, output_folder):
    """Saves a dedicated CSV for just Trial 1's results without rounding."""
    filename = os.path.join(output_folder, "trial_01_metrics.csv")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["trial_01_metrics.csv"])
        writer.writerow([])
        
        # Stoppage Times & Duration (No Rounding)
        st_steps = result['stop_ticks']
        st_time = result['stop_times']
        
        m_stop_steps = statistics.mean(st_steps) if st_steps else 0.0
        sd_stop_steps = statistics.stdev(st_steps) if len(st_steps) > 1 else 0.0
        m_stop_time = statistics.mean(st_time) if st_time else 0.0
        sd_stop_time = statistics.stdev(st_time) if len(st_time) > 1 else 0.0
        
        writer.writerow(["Duration (Steps)", result['duration_steps']])
        writer.writerow(["Duration (Video Seconds)", result['duration']])
        writer.writerow(["Mean Stop Time (steps)", m_stop_steps])
        writer.writerow(["SD of Stop Times (steps)", sd_stop_steps])
        writer.writerow(["Mean Stop Time (s)", m_stop_time])
        writer.writerow(["SD of Stop Times (s)", sd_stop_time])
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

def export_aggregate_summary(all_results, output_folder, final_stats):
    """Saves the master summary incorporating intra-run and pooled statistics."""
    filename = os.path.join(output_folder, "aggregate_summary.csv")

    # Flag any anomalous trials where the net displacement was backwards (negative)
    bad_rock_trials = [str(r['trial']) for r in all_results if r['mean_rock_displacement'] <= 0]
    bad_bbot_trials = [str(r['trial']) for r in all_results if r['mean_bbot_displacement'] <= 0]

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(["aggregate_summary.csv"])
        writer.writerow([])
        
        writer.writerow(["Trials with Mean Rock Displacement <= 0", len(bad_rock_trials)])
        writer.writerow(["Specific Rock Trial IDs", ", ".join(bad_rock_trials) if bad_rock_trials else "None"])
        writer.writerow([])
        writer.writerow(["Trials with Mean BBot Displacement <= 0", len(bad_bbot_trials)])
        writer.writerow(["Specific BBot Trial IDs", ", ".join(bad_bbot_trials) if bad_bbot_trials else "None"])
        writer.writerow([])
        
        writer.writerow(["Total Trials", len(all_results)])
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
        
        # Trial By Trial Breakdown (No Rounding)
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

def ui_get_simulation_parameters(screen):
    """Displays the main menu and captures keyboard input for trials."""
    clock = pygame.time.Clock()
    prompts = [
        "1. Enter number of trials:"
    ]
    inputs = [""]
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
            draw_text(screen, "BRAITENBERG VERSION ZERO BROWNIAN BOT (BBOT)", 20, WIDTH//2 - 260, HEIGHT//2 - 100, (255, 215, 0))
            draw_text(screen, prompts[step], 22, WIDTH//2 - 200, HEIGHT//2 - 20)
            draw_text(screen, inputs[step] + "_", 36, WIDTH//2 - 30, HEIGHT//2 + 30, (100, 255, 100))
            pygame.display.flip() 
            
        clock.tick(30) 
        
    num_trials = max(1, int(inputs[0])) if inputs[0] else 1
    
    return num_trials

def ui_show_headless_progress(screen, current, total):
    """Displays a loading screen while background simulations are crunching."""
    screen.fill(BG_COLOR)
    draw_text(screen, "RUNNING HEADLESS SIMULATIONS", 28, WIDTH//2 - 200, HEIGHT//2 - 40, (255, 215, 0))
    draw_text(screen, f"Crunching Trial {current} / {total}...", 20, WIDTH//2 - 100, HEIGHT//2 + 10)
    pygame.display.flip()
    pygame.event.pump() 

def ui_show_trial1_results(screen, duration_secs, duration_steps, mean_rock_displacement, sd_rock_displacement, mean_bbot_displacement, sd_bbot_displacement, output_folder, mean_stop_time, sd_stop_time, mean_stop_steps, sd_stop_steps):
    """Pauses the program after Trial 1 to show the user the baseline stats before doing headless runs."""
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
        
        # Steps Data above Seconds Data
        draw_text(screen, f"Mean Stop:      {mean_stop_steps:.2f} steps", 18, 30, 135)
        draw_text(screen, f"SD of Stops:    {sd_stop_steps:.2f} steps", 18, 30, 160, (200, 200, 200))
        draw_text(screen, f"Mean Stop:      {mean_stop_time:.2f} s", 18, 30, 200)
        draw_text(screen, f"SD of Stops:    {sd_stop_time:.2f} s", 18, 30, 225, (200, 200, 200))
        
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

def run_simulation(trial_num, is_visual, output_folder, screen=None):
    """Executes a single trial run of the physics simulation."""
    bbots = [BBot() for _ in range(NUM_BBOTS)]
    blocks = [Block(bbots) for _ in range(NUM_BLOCKS)] 
    
    ticks, sim_time, final_time = 0, 0.0, None
    video_writer = None # We will use this to export the MP4 video!
    
    # --- VIDEO & VISUAL SETUP FOR TRIAL 1 ---
    if is_visual and screen:
        pygame.display.set_caption("BRAITENBERG VERSION ZERO BROWNIAN BOT (BBOT)")
        
        # Initialize OpenCV Video Writer to export an MP4 at 60 Frames Per Second
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
        sim_time = ticks / 60.0  # Convert frames to real seconds
        
        check_collisions(bbots, blocks) 
        for b in blocks: b.update() 
        
        safe_bbots = 0 
        for bbot in bbots:
            bbot.sense(blocks, LIGHT_POS) 
            
            if bbot.in_shadow:
                bbot.is_stopped = True
            else:
                bbot.is_stopped = False
                
            bbot.decide_and_move() 
            
            if bbot.is_stopped:
                safe_bbots += 1
                if bbot.stop_tick is None:
                    bbot.stop_tick = ticks
                    bbot.stop_time = sim_time 
            else:
                bbot.stop_tick = None
                bbot.stop_time = None 
                
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
            pygame.draw.circle(screen, LIGHT_COLOR, LIGHT_POS, LIGHT_RADIUS_VISUAL)
            
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
        video_writer.release() # Stop recording the video and safely close the MP4 file!

    # Collect final math for this specific run
    rock_displacements = [get_distance_pixels(b.x, b.y) - get_distance_pixels(b.start_x, b.start_y) for b in blocks]
    mean_rock_displacement = statistics.mean(rock_displacements)
    
    bbot_displacements = [get_distance_pixels(bbot.x, bbot.y) - get_distance_pixels(bbot.start_x, bbot.start_y) for bbot in bbots]
    mean_bbot_displacement = statistics.mean(bbot_displacements)

    stop_ticks = [bbot.stop_tick if bbot.stop_tick is not None else ticks for bbot in bbots]
    stop_times = [bbot.stop_time if bbot.stop_time is not None else final_time for bbot in bbots]

    # Process final Trial 1 outputs
    if is_visual and screen:
        save_screenshot(screen, output_folder, "03_sim_finished.png")
        
        # --- NEW WAIT SCREEN: Forces the user to hit ENTER before showing Trial 1 stats ---
        waiting_for_stats = True
        clock = pygame.time.Clock()
        while waiting_for_stats:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN: 
                    waiting_for_stats = False

            screen.fill(BG_COLOR)
            draw_text(screen, "SIMULATION COMPLETE", 32, WIDTH//2 - 150, HEIGHT//2 - 40, (255, 215, 0))
            draw_text(screen, "Press [ENTER] to view Trial 1 stats.", 22, WIDTH//2 - 160, HEIGHT//2 + 20, (255, 100, 100))
            pygame.display.flip()
            clock.tick(15)
        # ----------------------------------------------------------------------------------
        
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
            m_stop_steps, sd_stop_steps
        )

    # Return the data packet to the master control loop
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
    pygame.display.set_caption("BRAITENBERG VERSION ZERO BROWNIAN BOT (BBOT)")

    num_trials = ui_get_simulation_parameters(screen)

    # Setup the output directory
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = f"simulation_output_{timestamp_str}"
    os.makedirs(output_folder, exist_ok=True)
    
    # Always run Trial 1 visually (This is the one that gets recorded as a video!)
    all_results = [run_simulation(1, True, output_folder, screen=screen)]
    
    # Export dedicated Trial 1 stats to CSV
    export_trial_01_metrics(all_results[0], output_folder)
    
    # Run the rest headlessly (no graphics drawn = extremely fast processing)
    if num_trials > 1:
        for i in range(2, num_trials + 1):
            if i % 10 == 0 or i == 2: 
                ui_show_headless_progress(screen, i, num_trials)
            all_results.append(run_simulation(i, False, output_folder))

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
        
        # Extend adds individual datapoints into one giant bucket (The Pooled Data)
        pooled_bbot_steps.extend(st_steps)
        pooled_bbot_times.extend(st_time)
        pooled_rock_displacements.extend(rd)
        pooled_bbot_displacements.extend(bd) 
        
        # Calculate the means for THIS specific run, and append to our intra-run tracker
        intra_run_rock_means.append(r['mean_rock_displacement'])
        intra_run_bbot_means.append(r['mean_bbot_displacement'])
        intra_run_step_means.append(statistics.mean(st_steps) if st_steps else 0.0)
        intra_run_time_means.append(statistics.mean(st_time) if st_time else 0.0)
            
    # --- CALCULATING THE REQUESTED METRICS ---
    # We package these into a dictionary to keep the code clean when sending to the export function.
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
    export_aggregate_summary(all_results, output_folder, final_stats)
                             
    plot_histogram(pooled_bbot_times, 'Distribution of BBot Stop Times', 'Simulation Time (seconds)', 'Frequency', "05_bbot_times_hist.png", 'royalblue', output_folder, True)
    plot_histogram(pooled_rock_displacements, 'Distribution of Net Rock Displacements', 'Net Displacement (pixels)', 'Frequency', "06_rock_displacement_hist.png", 'forestgreen', output_folder)
    plot_histogram(pooled_bbot_displacements, 'Distribution of Net BBot Displacements', 'Net Displacement (pixels)', 'Frequency', "07_bbot_displacement_hist.png", 'darkorange', output_folder)
    
    ui_show_final_done(screen, output_folder)
    pygame.quit()

if __name__ == "__main__":
    main()