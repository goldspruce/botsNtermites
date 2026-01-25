import pygame   # Import the library used for graphics and game loops
import math     # Import math library for calculating distances and angles
import random   # Import library to generate random numbers (for positions)
import sys      # Import system library to help exit the program cleanly
import time     # Import library to track how many seconds pass
import os       # Import operating system library to manage folders and files
from datetime import datetime # Import datetime to timestamp the folder

# =============================================================================
# SECTION 1: GLOBAL CONFIGURATION
# variables written in CAPS are constants (settings that don't change during the game)
# =============================================================================

# -- SCENARIO SETTINGS --
BLOCK_SPAWN_MODE = "EXCLUSION" 
ROBOT_SPAWN_MODE = "LIGHT_PERIMETER"

# -- WINDOW SETTINGS --
WIDTH, HEIGHT = 600, 600          # The width and height of the screen in pixels
BG_COLOR = (20, 20, 20)           # Dark Grey background color (Red, Green, Blue)
LIGHT_COLOR = (255, 255, 240)     # Bright White/Yellow for the light

# -- LIGHT POSITION --
LIGHT_POS = (30, 30)              # X, Y coordinate of the light (Top-Left)
LIGHT_RADIUS_VISUAL = 40          # How big the light bulb looks

# -- COLORS --
ROBOT_COLOR = (220, 60, 60)       # Red
REST_COLOR = (60, 220, 60)        # Green
BLOCK_COLOR = (80, 80, 220)       # Blue
WALL_COLOR = (255, 255, 0)        # Yellow
TEXT_COLOR = (255, 255, 255)      # White

# -- ENTITIES (The things in our simulation) --
NUM_ROBOTS = 20                   # We will create 20 robots
NUM_BLOCKS = 30                   # We will create 30 blocks
BLOCK_RADIUS = 13                 # Size if it's a Circle
BLOCK_SIZE = 26                   # Size if it's a Square
ROBOT_RADIUS = 10                 # Size of the robot
WALL_THICKNESS = 10               # How thick the border walls are

# -- LOGIC & PHYSICS --
MIN_SAFE_DISTANCE = 400           # Distance from light where shadows stop working (glare)
SHADOW_PROXIMITY_LIMIT = 20       # Robot must be this close to a block to trust the shadow
ROBOT_SPEED_GO = 3.5              # How fast robots move when Red
ROBOT_SPEED_PUSH = 1.0            # How fast robots move when pushing a block
PUSH_FORCE = 1.0                  # How hard they push
BLOCK_FRICTION = 0.90             # Friction makes blocks slow down after being pushed

# -- OUTPUT SETTINGS --
# Generate a timestamp string: Year-Month-Day_Hour-Minute
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
OUTPUT_FOLDER = f"simulation_output_{timestamp_str}"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created output folder: {OUTPUT_FOLDER}")

# Calculate the longest possible distance in the room (Corner to Corner)
MAX_POSSIBLE_DIST = math.hypot(WIDTH - LIGHT_POS[0], HEIGHT - LIGHT_POS[1])

# =============================================================================
# SECTION 2: HELPER FUNCTIONS
# "def" creates a new command (function) we can use later.
# =============================================================================

# Helper for saving images
def save_screenshot(screen, filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    pygame.image.save(screen, path)
    print(f"Screenshot saved: {path}")

# This function calculates a random starting position for a Robot.
def get_robot_spawn_pos():
    # Pick a random distance (r) slightly outside the light bulb
    r = random.uniform(LIGHT_RADIUS_VISUAL + 5, LIGHT_RADIUS_VISUAL + 25)
    # Pick a random angle (theta)
    theta = random.uniform(0, math.pi / 2)
    
    # Convert angle/distance to X and Y coordinates
    x = LIGHT_POS[0] + r * math.cos(theta)
    y = LIGHT_POS[1] + r * math.sin(theta)
    
    # Ensure the robot isn't inside a wall using min/max logic
    x = max(WALL_THICKNESS + ROBOT_RADIUS, min(WIDTH - WALL_THICKNESS - ROBOT_RADIUS, x))
    y = max(WALL_THICKNESS + ROBOT_RADIUS, min(HEIGHT - WALL_THICKNESS - ROBOT_RADIUS, y))
    
    return x, y # Send these coordinates back to whoever called this function

# This function finds a safe spot for a Block where it won't overlap a robot.
def get_valid_block_pos(existing_robots):
    # Calculate the safe area inside the walls
    safe_min = WALL_THICKNESS + BLOCK_RADIUS
    safe_max_x = WIDTH - WALL_THICKNESS - BLOCK_RADIUS
    safe_max_y = HEIGHT - WALL_THICKNESS - BLOCK_RADIUS

    while True: # Keep trying forever until we find a valid spot
        # Pick a random spot
        x = random.randint(int(safe_min), int(safe_max_x))
        y = random.randint(int(safe_min), int(safe_max_y))
        
        # Check 1: Is it too close to the light?
        dist_to_light = math.hypot(x - LIGHT_POS[0], y - LIGHT_POS[1])
        if dist_to_light < (LIGHT_RADIUS_VISUAL + BLOCK_RADIUS + 5):
            continue # If yes, skip the rest of the loop and try again
            
        # Check 2: Does it hit any existing robots?
        collision_found = False
        for r in existing_robots: # "r" is the current robot we are checking
            dist = math.hypot(x - r.x, y - r.y)
            if dist < (BLOCK_RADIUS + ROBOT_RADIUS + 2): 
                collision_found = True
                break
        
        if collision_found:
            continue # If collision found, try again

        # If we passed all checks, return the X and Y
        return x, y

# =============================================================================
# SECTION 3: CLASS DEFINITIONS
# Classes are "Blueprints". A class defines what data an object holds and what it can do.
# =============================================================================

class Robot:
    # __init__ is the setup function that runs automatically when we create a new Robot.
    def __init__(self):
        self.x, self.y = get_robot_spawn_pos() # Call our helper function to get start pos
        self.angle = random.uniform(0, 2 * math.pi) # Pick random facing direction
        self.touching_block = False 
        self.in_shadow = False

    # The "sense" method allows the robot to "look" at the world.
    def sense(self, blocks, light_pos):
        self.in_shadow = False
        self.touching_block = False
        
        # Calculate distance to light
        dx = self.x - light_pos[0]
        dy = self.y - light_pos[1]
        dist_to_light = math.hypot(dx, dy)
        
        if dist_to_light == 0: return 

        # If we are too close to the light (glare), shadows don't count
        if dist_to_light < MIN_SAFE_DISTANCE:
            self.in_shadow = False 
            return 

        # Math to draw a line from the light to the robot (Raycasting)
        dir_x = -(dx / dist_to_light)
        dir_y = -(dy / dist_to_light)
        sensor_x = self.x + (dir_x * ROBOT_RADIUS)
        sensor_y = self.y + (dir_y * ROBOT_RADIUS)
        
        line_start = light_pos
        line_end = (sensor_x, sensor_y)
        
        # Check if any block blocks that line
        for block in blocks:
            # Create a rectangle box around the block for math checking
            if block.shape == "CIRCLE":
                bx = block.x - block.radius
                by = block.y - block.radius
                size = block.radius * 2
            else: # SQUARE
                bx = block.x - (block.w / 2)
                by = block.y - (block.h / 2)
                size = block.w
            
            rect = pygame.Rect(int(bx), int(by), int(size), int(size))
            
            # "clipline" checks if the line crosses the rectangle
            if rect.clipline(line_start, line_end):
                # We found a shadow! But is the block close enough?
                dist_robot_block = math.hypot(self.x - block.x, self.y - block.y)
                
                required_dist = SHADOW_PROXIMITY_LIMIT + ROBOT_RADIUS + (block.radius if block.shape == "CIRCLE" else block.w/2)
                
                # Only trust the shadow if we are close
                if dist_robot_block <= required_dist:
                    self.in_shadow = True
                    break

    # The "decide_and_move" method controls behavior
    def decide_and_move(self):
        # Decision 1: If in shadow, do nothing (STOP)
        if self.in_shadow:
            return 
        
        # Decision 2: If pushing a block, move slow. If free, move fast.
        if self.touching_block:
            speed = ROBOT_SPEED_PUSH
        else:
            speed = ROBOT_SPEED_GO
            # Wiggle direction slightly to look like it's searching
            self.angle += random.uniform(-0.4, 0.4) 

        # Update Position (Physics math: converting Angle/Speed to X/Y motion)
        self.x += math.cos(self.angle) * speed
        self.y += math.sin(self.angle) * speed
        
        # Ensure we didn't walk through a wall
        self.force_bounds()

    # Helper method to keep robot inside the room
    def force_bounds(self):
        min_bound = WALL_THICKNESS + ROBOT_RADIUS
        max_x_bound = WIDTH - WALL_THICKNESS - ROBOT_RADIUS
        max_y_bound = HEIGHT - WALL_THICKNESS - ROBOT_RADIUS
        self.x = max(min_bound, min(max_x_bound, self.x))
        self.y = max(min_bound, min(max_y_bound, self.y))

class Block:
    # Setup the Block
    def __init__(self, existing_robots, shape_type):
        self.shape = shape_type # "CIRCLE" or "SQUARE"
        self.radius = BLOCK_RADIUS
        self.w = BLOCK_SIZE
        self.h = BLOCK_SIZE
        
        # Get a valid spawn point
        self.x, self.y = get_valid_block_pos(existing_robots)
        
        # Remember where we started (for the report at the end)
        self.start_x = self.x
        self.start_y = self.y
        
        # Velocity (Speed) variables
        self.vx = 0.0
        self.vy = 0.0

    # Physics update for the block
    def update(self):
        self.x += self.vx
        self.y += self.vy
        
        # Apply friction (slow down over time)
        self.vx *= BLOCK_FRICTION
        self.vy *= BLOCK_FRICTION
        
        # Keep block inside walls
        half_size = self.radius if self.shape == "CIRCLE" else (self.w / 2)
        min_bound = WALL_THICKNESS + half_size
        max_x_bound = WIDTH - WALL_THICKNESS - half_size
        max_y_bound = HEIGHT - WALL_THICKNESS - half_size

        if self.x < min_bound: self.x = min_bound; self.vx = 0
        if self.x > max_x_bound: self.x = max_x_bound; self.vx = 0
        if self.y < min_bound: self.y = min_bound; self.vy = 0
        if self.y > max_y_bound: self.y = max_y_bound; self.vy = 0

# =============================================================================
# SECTION 4: PHYSICS & REPORTING
# =============================================================================

# This function checks if things are bumping into each other
def check_collisions(robots, blocks):
    # --- Check Robot vs Block ---
    for robot in robots:
        for block in blocks:
            # Simplification: Treat everything as a circle for collision math
            eff_radius = block.radius if block.shape == "CIRCLE" else (block.w / 2)
            
            dx = block.x - robot.x
            dy = block.y - robot.y
            dist = math.hypot(dx, dy)
            
            # If distance is less than the two radii combined -> COLLISION
            if dist < (ROBOT_RADIUS + eff_radius):
                robot.touching_block = True
                if dist < 0.1: dist = 0.1 
                
                # Calculate push direction
                nx = dx / dist
                ny = dy / dist
                
                # Add speed to block (push it)
                block.vx += nx * PUSH_FORCE
                block.vy += ny * PUSH_FORCE
                
                # Push robot back slightly (Newton's 3rd law)
                robot.x -= nx * 2
                robot.y -= ny * 2

    # --- Check Block vs Block ---
    for i, b1 in enumerate(blocks):
        # Compare every block against every other block
        for b2 in blocks[i+1:]:
            dx = b1.x - b2.x
            dy = b1.y - b2.y
            dist = math.hypot(dx, dy)
            
            r1 = b1.radius if b1.shape == "CIRCLE" else (b1.w / 2)
            r2 = b2.radius if b2.shape == "CIRCLE" else (b2.w / 2)
            
            if dist < (r1 + r2):
                if dist < 0.1: dist = 0.1
                push = 2.0
                nx = dx / dist
                ny = dy / dist
                # Move them apart
                b1.x += nx * push; b1.y += ny * push
                b2.x -= nx * push; b2.y -= ny * push

# Helper to draw text on the screen
def draw_text(screen, text, size, x, y, color=TEXT_COLOR):
    font = pygame.font.SysFont("Arial", size, bold=False)
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))

# Helper to calculate % distance from light
def get_distance_score(x, y):
    dx = x - LIGHT_POS[0]
    dy = y - LIGHT_POS[1]
    dist = math.hypot(dx, dy)
    percent = (dist / MAX_POSSIBLE_DIST) * 100.0
    return percent

# The Final Report Screen function
def show_results_screen(screen, blocks, duration):
    # Calculate stats
    total_start_dist = 0
    total_end_dist = 0
    block_stats = []

    for i, b in enumerate(blocks):
        p_start = get_distance_score(b.start_x, b.start_y)
        p_end = get_distance_score(b.x, b.y)
        change = p_end - p_start
        total_start_dist += p_start
        total_end_dist += p_end
        block_stats.append(f"Block {i+1:02d}: {p_end:.1f}% ({change:+.1f}%)".replace('+', ''))

    avg_start = total_start_dist / len(blocks)
    avg_end = total_end_dist / len(blocks)
    avg_change = avg_end - avg_start
    
    screenshot_taken = False # <--- Flag to ensure we only save once in the loop

    # Loop to keep the window open until user closes it
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False

        screen.fill(BG_COLOR)
        draw_text(screen, "SIMULATION COMPLETE", 36, 30, 30, (255, 215, 0))
        
        draw_text(screen, f"Start AVG Dist:  {avg_start:.1f}%", 24, 30, 80)
        draw_text(screen, f"End AVG Dist:    {avg_end:.1f}%", 24, 30, 110)
        
        sign_str = "" if avg_change < 0 else ""
        draw_text(screen, f"Net Change:      {sign_str}{avg_change:.1f}%", 24, 30, 150, (200, 200, 200))
        
        # <--- CHANGED: RESOLVE TIME TO 1 SEC
        draw_text(screen, f"Time to Finish:  {int(duration)} sec", 24, 30, 190)
        
        if avg_change > 0:
            verdict = "AS EXPECTED: Pushed blocks away from entrance"
            v_color = REST_COLOR
        else:
            verdict = "NOT EXPECTED: Pushed blocks towards entrance"
            v_color = ROBOT_COLOR
        draw_text(screen, verdict, 24, 30, 230, v_color)

        draw_text(screen, "Individual Block Data:", 20, 30, 270, (150, 150, 150))
        
        # Display list of blocks
        y_start = 300
        line_height = 24
        for i in range(12):
            if i < len(block_stats):
                draw_text(screen, block_stats[i], 18, 30, y_start + (i * line_height))
        for i in range(12, len(block_stats)):
            draw_text(screen, block_stats[i], 18, 280, y_start + ((i-12) * line_height))

        draw_text(screen, "Press [Close Window] to exit.", 16, 30, HEIGHT - 30, (100, 100, 100))

        pygame.display.flip()
        
        # <--- SAVE FINAL SCREENSHOT (Metrics)
        if not screenshot_taken:
            save_screenshot(screen, "04_final_metrics.png")
            screenshot_taken = True
            
        pygame.time.Clock().tick(15)

# =============================================================================
# SECTION 5: MAIN EXECUTION
# This is where the program actually starts running.
# =============================================================================
def main():
    print(f"Initializing Simulation")
    pygame.init()                # Initialize the graphics system
    pygame.font.init()           # Initialize the font system
    screen = pygame.display.set_mode((WIDTH, HEIGHT)) # Create the window
    pygame.display.set_caption(f"Braitenberg Vehicle Zero: Brownian Bots")
    clock = pygame.time.Clock()  # Create a timer

    # -- SPAWN SETUP --
    # Create the List of Robots
    robots = [Robot() for _ in range(NUM_ROBOTS)]
    
    # Create the List of Blocks
    blocks = []
    for i in range(NUM_BLOCKS):
        # If 'i' is less than half the total, make it a CIRCLE, else SQUARE
        shape = "CIRCLE" if i < (NUM_BLOCKS // 2) else "SQUARE"
        blocks.append(Block(robots, shape))

    # Variables to track time
    start_time = None
    final_time = None 
    
    # <--- SCREENSHOT TIMERS
    last_screenshot_time = 0 
    start_screenshot_taken = False
    
    running = True            # Is the program window open?
    simulation_active = True  # Is the simulation mode on (vs report mode)?
    simulation_started = False # Has the user pressed ENTER yet?

    # -- THE MAIN LOOP --
    while running:
        # 1. Process User Input (Events)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # User clicked 'X' on window
                running = False
            if event.type == pygame.KEYDOWN:
                if not simulation_started:
                    if event.key == pygame.K_RETURN: # User pressed ENTER to start
                        simulation_started = True
                        start_time = time.time()
                        last_screenshot_time = start_time # Reset timer
                else:
                    if event.key == pygame.K_RETURN: # User pressed ENTER to end early
                        simulation_active = False 
                    if event.key == pygame.K_SPACE:  # User pressed SPACE (restart trick)
                        main() 
                        return

        if simulation_active:
            # 2. Run Game Logic (Only if started)
            if simulation_started:
                check_collisions(robots, blocks)
                for b in blocks: b.update() # Move blocks
                
                safe_robots = 0
                for r in robots:
                    r.sense(blocks, LIGHT_POS)   # Robot looks for shadows
                    r.decide_and_move()          # Robot moves
                    r.force_bounds()             # Robot stays in walls
                    if r.in_shadow:
                        safe_robots += 1
                
                # Check Victory Condition
                # if final_time is not None, the timer stops updating
                if safe_robots == NUM_ROBOTS and final_time is None:
                    final_time = time.time() - start_time
            
            # 3. Draw Everything
            screen.fill(BG_COLOR) # Clear screen
            
            # Draw Zones (Visual only)
            pygame.draw.circle(screen, (30, 30, 30), LIGHT_POS, MIN_SAFE_DISTANCE)
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, HEIGHT-WALL_THICKNESS, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WALL_THICKNESS, HEIGHT))
            pygame.draw.rect(screen, WALL_COLOR, (WIDTH-WALL_THICKNESS, 0, WALL_THICKNESS, HEIGHT))

            pygame.draw.circle(screen, LIGHT_COLOR, LIGHT_POS, LIGHT_RADIUS_VISUAL)
            
            # <--- DRAW TIMER NEAR LIGHT
            if simulation_started:
                # If robots are done (final_time set), freeze timer. Otherwise show elapsed.
                if final_time is not None:
                    time_to_show = final_time
                else:
                    time_to_show = time.time() - start_time
                
                # CHANGED: Use int() to resolve to 1 sec
                draw_text(screen, f"{int(time_to_show)}s", 20, 10, 80, (255, 255, 0))

            # Draw Blocks
            for b in blocks: 
                if b.shape == "CIRCLE":
                    pygame.draw.circle(screen, BLOCK_COLOR, (int(b.x), int(b.y)), int(b.radius))
                else:
                    # Squares are drawn from top-left, so we adjust from center
                    left = int(b.x - b.w/2)
                    top = int(b.y - b.h/2)
                    pygame.draw.rect(screen, BLOCK_COLOR, (left, top, b.w, b.h))
            
            # Draw Robots
            for r in robots:
                color = REST_COLOR if r.in_shadow else ROBOT_COLOR
                pygame.draw.circle(screen, color, (int(r.x), int(r.y)), ROBOT_RADIUS)
                # Draw the little white "eye" on the robot
                eye_x = r.x + math.cos(r.angle) * (ROBOT_RADIUS - 5)
                eye_y = r.y + math.sin(r.angle) * (ROBOT_RADIUS - 5)
                pygame.draw.circle(screen, (255, 255, 255), (int(eye_x), int(eye_y)), 3)

            # Draw Text overlays
            if not simulation_started:
                draw_text(screen, "Press ENTER to Start", 32, WIDTH//2 - 120, 100, (255, 255, 0))
            else:
                draw_text(screen, f"Glare: 400px", 16, 20, HEIGHT - 50, (150, 150, 150))
                draw_text(screen, "Press ENTER for Metrics", 16, 20, HEIGHT - 30)

            # Flip the display (update screen)
            pygame.display.flip()
            
            # <--- SCREENSHOT LOGIC (During Simulation)
            
            # 1. Start Screen (Once)
            if simulation_started and not start_screenshot_taken:
                save_screenshot(screen, "01_sim_start.png")
                start_screenshot_taken = True
                
            # 2. Every 5 Seconds
            if simulation_started:
                current_time = time.time()
                if current_time - last_screenshot_time >= 5:
                    timestamp = int(current_time - start_time)
                    save_screenshot(screen, f"02_progress_{timestamp}s.png")
                    last_screenshot_time = current_time

            clock.tick(60) # Limit to 60 Frames Per Second
        
        else:
            # <--- SCREENSHOT LOGIC (End of Simulation)
            # Save the state of robots/blocks one last time before switching to report
            save_screenshot(screen, "03_sim_finished.png")
            
            # If simulation_active is False, show the report
            if final_time is not None:
                duration_to_show = final_time
            else:
                duration_to_show = time.time() - start_time
            show_results_screen(screen, blocks, duration_to_show)
            running = False 

    # Clean up when loop finishes
    pygame.quit()
    sys.exit()

# This check ensures the code only runs if executed directly (not imported)
if __name__ == "__main__":
    main()