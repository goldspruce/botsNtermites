import pygame   # The main library for graphics and window handling
import math     # Standard math functions (sqrt, cos, sin, hypot)
import random   # To generate random positions and angles
import sys      # To interact with the system (exit the program)
import time     # To track the duration of the simulation

# =============================================================================
# SECTION 1: GLOBAL CONFIGURATION (CONSTANTS)
# These variables control the "Rules of the Universe" for this simulation.
# =============================================================================

# -- WINDOW SETTINGS --
WIDTH, HEIGHT = 600, 600          # The size of the simulation window in pixels
BG_COLOR = (20, 20, 20)           # Dark Grey background
LIGHT_COLOR = (255, 255, 240)     # Warm White for the central light
# The light is placed exactly in the center of the screen
LIGHT_POS = (WIDTH // 2, HEIGHT // 2) 

# -- COLOR PALETTE (R, G, B) --
ROBOT_COLOR = (220, 60, 60)       # Red (Danger/Panic state)
REST_COLOR = (60, 220, 60)        # Green (Safe/Resting state)
BLOCK_COLOR = (80, 80, 220)       # Blue (The obstacles)
WALL_COLOR = (255, 255, 0)        # Yellow (The boundaries)
TEXT_COLOR = (255, 255, 255)      # White (Text output)

# -- SIMULATION ENTITIES --
NUM_ROBOTS = 20                   # Total number of agents
NUM_BLOCKS = 30                   # Total number of obstacles
BLOCK_SIZE = 30                   # Size of the square blocks (30x30 px)
ROBOT_RADIUS = 10                 # Size of the circular robots
WALL_THICKNESS = 10               # Thickness of the yellow border walls

# -- LOGIC THRESHOLDS --
MIN_SAFE_DISTANCE = 350           # The "Glare Zone". Inside this radius, it is
                                  # impossible to find a shadow because the light 
                                  # is too pervasive/ambient.

# -- PHYSICS PARAMETERS --
ROBOT_SPEED_PANIC = 3.5           # High speed when exposed to light
ROBOT_SPEED_PUSH = 1.0            # Low speed when pushing an object (Simulates weight)
PUSH_FORCE = 1.0                  # How much velocity a robot transfers to a block
BLOCK_FRICTION = 0.90             # Velocity decay. Blocks lose 10% speed per frame.
                                  # This prevents them from sliding like on ice.

# -- METRIC CALCULATION --
# To calculate "Peripherality" (0-100%), we need to know the maximum possible
# distance from the center. In a rectangle, this is the distance to a corner.
# Pythagorean theorem: a^2 + b^2 = c^2
# max_dist = sqrt( (width/2)^2 + (height/2)^2 )
MAX_POSSIBLE_DIST = math.hypot(WIDTH // 2, HEIGHT // 2)

# =============================================================================
# SECTION 2: CLASS DEFINITIONS
# =============================================================================

class Robot:
    """
    The Agent. It has simple behavior:
    1. If in Light -> Move Fast & Randomly (Kinesis/Panic).
    2. If in Shadow -> Stop Moving.
    """
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.angle = random.uniform(0, 2 * math.pi) # Random starting direction
        
        # Flags to track state
        self.touching_block = False 
        self.in_shadow = False

    def sense(self, blocks, light_pos):
        """
        Determines if the robot is safe (in shadow) or exposed (in light).
        Uses Raycasting logic.
        """
        # Reset state for this frame
        self.in_shadow = False
        self.touching_block = False
        
        # Vector Math: Vector from Light to Robot
        dx = self.x - light_pos[0]
        dy = self.y - light_pos[1]
        dist = math.hypot(dx, dy)
        
        # Error handling: Avoid divide by zero
        if dist == 0: return 

        # 1. GLARE CHECK
        # If the robot is too close to the light source, shadows don't count.
        # This forces robots to push blocks further away to find safety.
        if dist < MIN_SAFE_DISTANCE:
            self.in_shadow = False 
            return 

        # 2. RAYCASTING (Line of Sight)
        # Calculate the direction unit vector pointing FROM light TO robot
        dir_x = -(dx / dist)
        dir_y = -(dy / dist)
        
        # We check a point slightly inside the robot (the "sensor" on its head)
        sensor_x = self.x + (dir_x * ROBOT_RADIUS)
        sensor_y = self.y + (dir_y * ROBOT_RADIUS)
        
        # Define the line segment: Light Source -> Robot Sensor
        line_start = light_pos
        line_end = (sensor_x, sensor_y)
        
        # Check if this line intersects ANY block
        for block in blocks:
            rect = pygame.Rect(int(block.x), int(block.y), block.w, block.h)
            if rect.clipline(line_start, line_end):
                self.in_shadow = True
                break # Found a shadow, no need to check other blocks

    def decide_and_move(self):
        """
        The 'Brain' of the robot. 
        Updates position based on state (Panic vs Rest).
        """
        # RULE 1: If safe, freeze. (Stigmergy: React to the environment)
        if self.in_shadow:
            return 
            
        # RULE 2: If pushing a heavy block, move slowly.
        if self.touching_block:
            speed = ROBOT_SPEED_PUSH
        
        # RULE 3: If exposed and free, PANIC!
        else:
            speed = ROBOT_SPEED_PANIC
            # "Brownian Motion": Add random jitter to direction
            self.angle += random.uniform(-0.4, 0.4)

        # Convert Polar coordinates (Angle/Speed) to Cartesian (X/Y)
        self.x += math.cos(self.angle) * speed
        self.y += math.sin(self.angle) * speed
        
        # Keep inside the walls
        self.force_bounds()

    def force_bounds(self):
        """ Hard clamp to keep robot inside the simulation window """
        min_bound = WALL_THICKNESS + ROBOT_RADIUS
        max_x_bound = WIDTH - WALL_THICKNESS - ROBOT_RADIUS
        max_y_bound = HEIGHT - WALL_THICKNESS - ROBOT_RADIUS
        self.x = max(min_bound, min(max_x_bound, self.x))
        self.y = max(min_bound, min(max_y_bound, self.y))

class Block:
    """
    The passive object. Robots push these.
    Stores its start position to calculate final stats.
    """
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        
        # DATA METRICS: Remember starting pos for comparison
        self.start_x = float(x)
        self.start_y = float(y)
        
        self.w = BLOCK_SIZE
        self.h = BLOCK_SIZE
        
        # Physics: Velocity (Speed)
        self.vx = 0.0
        self.vy = 0.0

    def update(self):
        """ Apply physics: Movement, Friction, Wall collisions """
        # Apply velocity
        self.x += self.vx
        self.y += self.vy
        
        # Apply Friction (Decay velocity so it stops eventually)
        self.vx *= BLOCK_FRICTION
        self.vy *= BLOCK_FRICTION
        
        # Wall Collisions (Stop if hitting edge)
        min_bound = WALL_THICKNESS
        max_x_bound = WIDTH - WALL_THICKNESS - self.w
        max_y_bound = HEIGHT - WALL_THICKNESS - self.h

        if self.x < min_bound: self.x = min_bound; self.vx = 0
        if self.x > max_x_bound: self.x = max_x_bound; self.vx = 0
        if self.y < min_bound: self.y = min_bound; self.vy = 0
        if self.y > max_y_bound: self.y = max_y_bound; self.vy = 0

# =============================================================================
# SECTION 3: PHYSICS ENGINE
# =============================================================================
def check_collisions(robots, blocks):
    """
    Handles interactions between:
    1. Robots & Blocks (Pushing)
    2. Blocks & Blocks (Stacking/Bumping)
    """
    
    # 1. ROBOT vs BLOCK
    for robot in robots:
        # Create hitboxes
        r_rect = pygame.Rect(int(robot.x - ROBOT_RADIUS), int(robot.y - ROBOT_RADIUS), 
                             ROBOT_RADIUS*2, ROBOT_RADIUS*2)
        
        for block in blocks:
            b_rect = pygame.Rect(int(block.x), int(block.y), block.w, block.h)
            
            if r_rect.colliderect(b_rect):
                robot.touching_block = True
                
                # Calculate push vector (Direction from Robot center to Block center)
                dx = (block.x + block.w/2) - robot.x
                dy = (block.y + block.h/2) - robot.y
                dist = math.hypot(dx, dy)
                
                # Normalize vector and apply force
                if dist < 0.1: dist = 1.0 
                nx = dx/dist
                ny = dy/dist
                
                # Transfer momentum: Block gains speed, Robot bounces back
                block.vx += nx * PUSH_FORCE
                block.vy += ny * PUSH_FORCE
                robot.x -= nx * 2
                robot.y -= ny * 2

    # 2. BLOCK vs BLOCK (Simple elastic collision)
    for i, b1 in enumerate(blocks):
        for b2 in blocks[i+1:]:
            r1 = pygame.Rect(int(b1.x), int(b1.y), b1.w, b1.h)
            r2 = pygame.Rect(int(b2.x), int(b2.y), b2.w, b2.h)
            
            if r1.colliderect(r2):
                # Push them apart based on their centers
                dx = b1.x - b2.x
                dy = b1.y - b2.y
                dist = math.hypot(dx, dy)
                if dist < 0.1: dist = 1.0
                
                push = 2.0 # Force strength
                b1.x += (dx/dist)*push; b1.y += (dy/dist)*push
                b2.x -= (dx/dist)*push; b2.y -= (dy/dist)*push

# =============================================================================
# SECTION 4: DISPLAY & METRICS
# =============================================================================

def draw_text(screen, text, size, x, y, color=TEXT_COLOR):
    """ Renders text using the system default Arial font """
    font = pygame.font.SysFont("Arial", size, bold=False)
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))

def get_peripherality_score(x, y):
    """
    Returns a score from 0.0 to 100.0 based on how far 
    coords (x,y) are from the center.
    """
    dx = x - LIGHT_POS[0]
    dy = y - LIGHT_POS[1]
    dist = math.hypot(dx, dy)
    
    # Calculate percentage of maximum possible distance
    percent = (dist / MAX_POSSIBLE_DIST) * 100.0
    return percent

def show_results_screen(screen, blocks, duration):
    """
    The Final Report Screen.
    Calculates and displays Peripherality metrics.
    """
    total_start_periph = 0
    total_end_periph = 0
    block_stats = [] # Will store strings for display

    # -- DATA ANALYSIS LOOP --
    for i, b in enumerate(blocks):
        # 1. Calculate Start/End Peripherality
        p_start = get_peripherality_score(b.start_x, b.start_y)
        p_end = get_peripherality_score(b.x, b.y)
        
        # 2. Calculate Change
        change = p_end - p_start
        
        # 3. Add to totals for average calculation
        total_start_periph += p_start
        total_end_periph += p_end
        
        # 4. Format String (e.g., "Block 01: 15.2% (+5.1%)")
        # Note: We omit the '+' sign for positive numbers as requested,
        # but negative numbers will automatically have '-'.
        block_stats.append(f"Block {i+1:02d}: {p_end:.1f}% ({change:+.1f}%)".replace('+', ''))

    # Calculate Global Averages
    avg_start = total_start_periph / len(blocks)
    avg_end = total_end_periph / len(blocks)
    avg_change = avg_end - avg_start

    # -- RENDER LOOP (Waits for user to close) --
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False

        screen.fill(BG_COLOR) # Clear screen

        # -- DRAW HEADERS --
        draw_text(screen, "SIMULATION COMPLETE", 36, 30, 30, (255, 215, 0)) # Gold Title
        
        # -- GLOBAL METRICS --
        # We display the total improvement in peripherality
        draw_text(screen, f"Avg Peripherality: {avg_end:.1f}%", 28, 30, 80)
        
        # Show the CHANGE in score
        sign_str = "" if avg_change < 0 else "" # No plus sign
        draw_text(screen, f"Net Change:        {sign_str}{avg_change:.1f}%", 24, 30, 120, (200, 200, 200))
        
        draw_text(screen, f"Time to Finish:    {duration:.1f} sec", 24, 30, 160)
        
        # -- VERDICT --
        if avg_change > 0:
            verdict = "VERDICT: SUCCESS (Pushed Outward)"
            v_color = REST_COLOR
        else:
            verdict = "VERDICT: FAILURE (Pulled Inward)"
            v_color = ROBOT_COLOR
        draw_text(screen, verdict, 24, 30, 200, v_color)

        # -- DATA TABLE --
        draw_text(screen, "Individual Block Peripherality (Change):", 20, 30, 240, (150, 150, 150))
        
        y_start = 270
        line_height = 24
        
        # Column 1 (0-12)
        for i in range(13):
            if i < len(block_stats):
                draw_text(screen, block_stats[i], 18, 30, y_start + (i * line_height))

        # Column 2 (13-24)
        for i in range(13, len(block_stats)):
            draw_text(screen, block_stats[i], 18, 280, y_start + ((i-13) * line_height))

        # Footer
        draw_text(screen, "Press [Close Window] to exit.", 16, 30, HEIGHT - 30, (100, 100, 100))

        pygame.display.flip()
        pygame.time.Clock().tick(15)

# =============================================================================
# SECTION 5: MAIN EXECUTION FLOW
# =============================================================================
def main():
    print("Initializing Simulation...")
    pygame.init()
    pygame.font.init() 
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Type I Swarm: Peripherality Metric")
    clock = pygame.time.Clock()

    # -- SPAWN SETUP --
    # Define a safe square inside the walls to spawn items
    safe_min = WALL_THICKNESS + 20
    safe_max_x = WIDTH - WALL_THICKNESS - 20
    safe_max_y = HEIGHT - WALL_THICKNESS - 20

    # Create Lists of Objects (List Comprehensions)
    robots = [Robot(random.randint(safe_min, safe_max_x), random.randint(safe_min, safe_max_y)) 
              for _ in range(NUM_ROBOTS)]
    blocks = [Block(random.randint(safe_min, safe_max_x), random.randint(safe_min, safe_max_y)) 
              for _ in range(NUM_BLOCKS)]

    # -- TIMER SETUP --
    start_time = time.time() 
    final_time = None 
    
    running = True
    simulation_active = True 

    # -- MAIN LOOP --
    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN: 
                    simulation_active = False # End sim, show stats
                if event.key == pygame.K_SPACE:
                    main() # Restart
                    return

        if simulation_active:
            # --- PHASE 1: UPDATE ---
            check_collisions(robots, blocks)
            for b in blocks: b.update()
            
            # Count how many robots have found safety
            safe_robots = 0
            for r in robots:
                r.sense(blocks, LIGHT_POS)
                r.decide_and_move()
                r.force_bounds()
                if r.in_shadow:
                    safe_robots += 1
            
            # --- TIMER LOGIC ---
            # Stop the timer the moment ALL robots are safe.
            # We check "if final_time is None" to ensure we only capture
            # the FIRST moment of success, not subsequent frames.
            if safe_robots == NUM_ROBOTS and final_time is None:
                final_time = time.time() - start_time
            
            # --- PHASE 2: DRAW ---
            screen.fill(BG_COLOR)
            
            # Draw Glare Zone
            pygame.draw.circle(screen, (30, 30, 30), LIGHT_POS, MIN_SAFE_DISTANCE)
            
            # Draw Walls
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, HEIGHT-WALL_THICKNESS, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WALL_THICKNESS, HEIGHT))
            pygame.draw.rect(screen, WALL_COLOR, (WIDTH-WALL_THICKNESS, 0, WALL_THICKNESS, HEIGHT))

            # Draw Light
            pygame.draw.circle(screen, LIGHT_COLOR, LIGHT_POS, 40)
            
            # Draw Blocks
            for b in blocks: 
                pygame.draw.rect(screen, BLOCK_COLOR, (int(b.x), int(b.y), b.w, b.h))
            
            # Draw Robots
            for r in robots:
                color = REST_COLOR if r.in_shadow else ROBOT_COLOR
                pygame.draw.circle(screen, color, (int(r.x), int(r.y)), ROBOT_RADIUS)
                # Draw Eyes (Direction Indicator)
                eye_x = r.x + math.cos(r.angle) * (ROBOT_RADIUS - 5)
                eye_y = r.y + math.sin(r.angle) * (ROBOT_RADIUS - 5)
                pygame.draw.circle(screen, (255, 255, 255), (int(eye_x), int(eye_y)), 3)

            # Overlay Text
            draw_text(screen, "Press ENTER for Metrics", 16, 20, HEIGHT - 30)

            pygame.display.flip()
            clock.tick(60) # Cap at 60 FPS
        
        else:
            # --- PHASE 3: RESULTS ---
            # Determine correct duration to display
            if final_time is not None:
                duration_to_show = final_time
            else:
                duration_to_show = time.time() - start_time
                
            show_results_screen(screen, blocks, duration_to_show)
            running = False # Exit app after closing results window

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()