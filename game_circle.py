import pygame   # Game engine and graphics
import math     # Vector math (hypot, sin, cos)
import random   # Stochasticity (Random noise/jitter)
import sys      # System exit
import time     # Duration tracking

# =============================================================================
# SECTION 1: GLOBAL CONFIGURATION ("The Environment")
# =============================================================================

WIDTH, HEIGHT = 600, 600          
BG_COLOR = (20, 20, 20)           
LIGHT_COLOR = (255, 255, 240)     
LIGHT_POS = (WIDTH // 2, HEIGHT // 2)

# -- VISUALS --
# We use distinctly different colors to visualize state
ROBOT_COLOR = (220, 60, 60)       # Red = High Energy / Panic
REST_COLOR = (60, 220, 60)        # Green = Low Energy / Rest
BLOCK_COLOR = (80, 80, 220)       # Blue = Passive Matter
WALL_COLOR = (255, 255, 0)        # Yellow = Hard Constraints
TEXT_COLOR = (255, 255, 255)      

# -- SIMULATION ENTITIES (User Defined) --
NUM_ROBOTS = 20                   # Total number of agents
NUM_BLOCKS = 30                   # Increased density (30 blocks)
BLOCK_SIZE = 30                   # Diameter of the blocks
BLOCK_RADIUS = BLOCK_SIZE / 2     # Radius for collision math (15.0)
ROBOT_RADIUS = 10                 # Radius of robots
WALL_THICKNESS = 10               

# -- LOGIC THRESHOLDS --
# The "Glare Zone" is now 350px.
# Since the distance from Center to Corner is approx 424px, 
# this leaves only a ~74px safe zone in the deep corners.
MIN_SAFE_DISTANCE = 350           

# -- PHYSICS PARAMETERS --
ROBOT_SPEED_PANIC = 3.5           # Speed when "fleeing" the light
ROBOT_SPEED_PUSH = 1.0            # Speed when blocked (simulates mass)
PUSH_FORCE = 0.8                  # Force transferred to blocks
BLOCK_FRICTION = 0.92             # Blocks slide slightly more (fluidity)

# Metric: Max distance from center to corner (approx 424.26 px)
MAX_POSSIBLE_DIST = math.hypot(WIDTH // 2, HEIGHT // 2)

# =============================================================================
# SECTION 2: THE AGENTS (Structural Coupling)
# =============================================================================

class Robot:
    """
    The Enactive Agent.
    It does not 'know' the room. It only knows its current sensation 
    (Light vs Dark) and reacts immediately (Panic vs Stop).
    """
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.angle = random.uniform(0, 2 * math.pi)
        self.touching_block = False 
        self.in_shadow = False

    def sense(self, blocks, light_pos):
        """
        Perception is not a camera image; it is a check for viability.
        'Am I safe?' (In Shadow) vs 'Am I burning?' (In Light).
        """
        self.in_shadow = False
        self.touching_block = False
        
        # Calculate distance to Light Source
        dx = self.x - light_pos[0]
        dy = self.y - light_pos[1]
        dist = math.hypot(dx, dy)
        
        if dist == 0: return 

        # 1. GLARE CHECK (The Environmental Constraint)
        # If too close to the light, shadows are too weak to hide in.
        if dist < MIN_SAFE_DISTANCE:
            self.in_shadow = False 
            return 

        # 2. RAYCASTING (Shadow Detection)
        # We check if a line from Light -> Robot hits any Block.
        # Uses a Bounding Box approximation for shadow-casting to keep FPS high.
        
        dir_x = -(dx / dist)
        dir_y = -(dy / dist)
        sensor_x = self.x + (dir_x * ROBOT_RADIUS)
        sensor_y = self.y + (dir_y * ROBOT_RADIUS)
        
        line_start = light_pos
        line_end = (sensor_x, sensor_y)
        
        for block in blocks:
            # We use a rectangular hitbox for the *shadow* calculation
            # This is a standard game-dev approximation.
            hitbox = pygame.Rect(int(block.x - BLOCK_RADIUS), int(block.y - BLOCK_RADIUS), 
                                 BLOCK_SIZE, BLOCK_SIZE)
            if hitbox.clipline(line_start, line_end):
                self.in_shadow = True
                break

    def decide_and_move(self):
        """
        Action.
        The robot maintains viability by stopping when safe.
        It lays down a path by panicking when unsafe.
        """
        # STATE 1: Homeostasis (Rest)
        if self.in_shadow:
            return 
            
        # STATE 2: Struggle (Pushing)
        if self.touching_block:
            speed = ROBOT_SPEED_PUSH
        
        # STATE 3: Panic (Kinesis)
        else:
            speed = ROBOT_SPEED_PANIC
            # Random Jitter: This prevents "Buridan's Ass" scenarios
            # where a robot gets stuck in a perfect equilibrium.
            self.angle += random.uniform(-0.4, 0.4)

        # Update Position
        self.x += math.cos(self.angle) * speed
        self.y += math.sin(self.angle) * speed
        
        self.force_bounds()

    def force_bounds(self):
        # Keep inside walls
        min_bound = WALL_THICKNESS + ROBOT_RADIUS
        max_x = WIDTH - WALL_THICKNESS - ROBOT_RADIUS
        max_y = HEIGHT - WALL_THICKNESS - ROBOT_RADIUS
        self.x = max(min_bound, min(max_x, self.x))
        self.y = max(min_bound, min(max_y, self.y))

class Block:
    """
    The Passive Environment.
    Now Circular to allow for hexagonal packing (Flow vs Stacking).
    """
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.radius = BLOCK_RADIUS
        
        # Metrics tracking
        self.start_x = float(x)
        self.start_y = float(y)
        
        # Velocity
        self.vx = 0.0
        self.vy = 0.0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= BLOCK_FRICTION
        self.vy *= BLOCK_FRICTION
        
        # Wall Collisions (Bounce off walls)
        min_b = WALL_THICKNESS + self.radius
        max_x = WIDTH - WALL_THICKNESS - self.radius
        max_y = HEIGHT - WALL_THICKNESS - self.radius

        if self.x < min_b: self.x = min_b; self.vx *= -0.5
        if self.x > max_x: self.x = max_x; self.vx *= -0.5
        if self.y < min_b: self.y = min_b; self.vy *= -0.5
        if self.y > max_y: self.y = max_y; self.vy *= -0.5

# =============================================================================
# SECTION 3: PHYSICS ENGINE (Circle-Circle Collision)
# =============================================================================

def check_collisions(robots, blocks):
    """
    Handles the physical interactions.
    Now uses DISTANCE CHECKS instead of RECTANGLES.
    This creates 'slippery' geometry that slides into corners.
    """
    
    # 1. ROBOT vs BLOCK (Circle vs Circle)
    for robot in robots:
        for block in blocks:
            dx = block.x - robot.x
            dy = block.y - robot.y
            dist = math.hypot(dx, dy)
            
            # Minimum allowed distance (Sum of radii)
            min_dist = ROBOT_RADIUS + block.radius
            
            if dist < min_dist:
                robot.touching_block = True
                
                # Collision Normal (Direction of push)
                if dist == 0: dist = 0.01
                nx = dx / dist
                ny = dy / dist
                
                # Transfer Energy to Block
                block.vx += nx * PUSH_FORCE
                block.vy += ny * PUSH_FORCE
                
                # Push Robot back (Action/Reaction)
                overlap = min_dist - dist
                robot.x -= nx * overlap
                robot.y -= ny * overlap

    # 2. BLOCK vs BLOCK (Circle vs Circle)
    # This is critical for the "Hexagonal Packing" effect.
    for i, b1 in enumerate(blocks):
        for b2 in blocks[i+1:]:
            dx = b1.x - b2.x
            dy = b1.y - b2.y
            dist = math.hypot(dx, dy)
            
            min_dist = b1.radius + b2.radius
            
            if dist < min_dist:
                # Resolve Overlap (Nudge them apart)
                if dist == 0: dist = 0.01
                overlap = min_dist - dist
                nx = dx / dist
                ny = dy / dist
                
                # Move each block away by half the overlap
                # This prevents "Stacking" and encourages "Sliding"
                b1.x += nx * (overlap * 0.5)
                b1.y += ny * (overlap * 0.5)
                b2.x -= nx * (overlap * 0.5)
                b2.y -= ny * (overlap * 0.5)
                
                # Transfer minor velocity (Simulation of energy transfer)
                avg_vx = (b1.vx + b2.vx) / 2
                avg_vy = (b1.vy + b2.vy) / 2
                b1.vx = avg_vx
                b2.vx = avg_vx # Simple inelastic transfer

# =============================================================================
# SECTION 4: METRICS & REPORTING
# =============================================================================

def draw_text(screen, text, size, x, y, color=TEXT_COLOR):
    font = pygame.font.SysFont("Arial", size, bold=False)
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))

def get_peripherality(x, y):
    """ Returns 0-100% score based on distance from center """
    dx = x - LIGHT_POS[0]
    dy = y - LIGHT_POS[1]
    dist = math.hypot(dx, dy)
    return (dist / MAX_POSSIBLE_DIST) * 100.0

def show_results(screen, blocks, duration):
    """ Displays final Peripherality stats with cleaned numbers """
    total_start, total_end = 0, 0
    stats = []

    for i, b in enumerate(blocks):
        p_start = get_peripherality(b.start_x, b.start_y)
        p_end = get_peripherality(b.x, b.y)
        change = p_end - p_start
        total_start += p_start
        total_end += p_end
        
        # Format: Remove '+' sign, keep '-' sign automatically
        stats.append(f"Block {i+1:02d}: {p_end:.1f}% ({change:+.1f}%)".replace('+', ''))

    avg_end = total_end / len(blocks)
    net_change = avg_end - (total_start / len(blocks))

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: waiting = False
        
        screen.fill(BG_COLOR)
        draw_text(screen, "SIMULATION COMPLETE", 36, 30, 30, (255, 215, 0))
        
        # Highlight: Total Displacement (Peripherality) at top
        draw_text(screen, f"Final Peripherality: {avg_end:.1f}%", 28, 30, 80)
        draw_text(screen, f"Net Change:          {net_change:+.1f}%".replace('+',''), 24, 30, 120, (200,200,200))
        draw_text(screen, f"Time to Finish:      {duration:.1f} sec", 24, 30, 160)

        # Verdict
        if net_change > 0:
            draw_text(screen, "VERDICT: SUCCESS (Swarm Cleared Area)", 24, 30, 200, REST_COLOR)
        else:
            draw_text(screen, "VERDICT: FAILURE (Area Cluttered)", 24, 30, 200, ROBOT_COLOR)

        # Columns
        draw_text(screen, "Individual Block Peripherality:", 20, 30, 240, (150,150,150))
        y_base = 270
        
        # Split into 2 columns for readability
        for i in range(15):
            if i < len(stats): draw_text(screen, stats[i], 18, 30, y_base + (i*24))
        for i in range(15, len(stats)):
            draw_text(screen, stats[i], 18, 250, y_base + ((i-15)*24))

        draw_text(screen, "Press [Close Window] to exit.", 16, 30, HEIGHT-30, (100,100,100))
        pygame.display.flip()
        pygame.time.Clock().tick(15)

# =============================================================================
# SECTION 5: MAIN LOOP
# =============================================================================
def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Enactive Swarm: Circular Physics")
    clock = pygame.time.Clock()

    # Safe Spawn Zone (Center area is dangerous, so we spawn randomly)
    # We spawn robots/blocks somewhat centrally to force them to work.
    spawn_min = WALL_THICKNESS + 20
    spawn_max = WIDTH - WALL_THICKNESS - 20

    robots = [Robot(random.randint(spawn_min, spawn_max), random.randint(spawn_min, spawn_max)) 
              for _ in range(NUM_ROBOTS)]
    
    # Initialize Circular Blocks
    blocks = [Block(random.randint(spawn_min, spawn_max), random.randint(spawn_min, spawn_max)) 
              for _ in range(NUM_BLOCKS)]

    start_time = time.time()
    final_time = None
    running = True
    simulation_active = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN: simulation_active = False
                if event.key == pygame.K_SPACE: main(); return

        if simulation_active:
            # 1. PHYSICS
            check_collisions(robots, blocks)
            for b in blocks: b.update()

            # 2. AGENT BEHAVIOR (Stigmergy)
            safe_count = 0
            for r in robots:
                r.sense(blocks, LIGHT_POS)
                r.decide_and_move()
                if r.in_shadow: safe_count += 1
            
            # 3. TIMER LOGIC (Freeze time when all safe)
            if safe_count == NUM_ROBOTS and final_time is None:
                final_time = time.time() - start_time

            # 4. DRAWING
            screen.fill(BG_COLOR)
            
            # Draw Danger Zone (Visual guide for the 350px limit)
            pygame.draw.circle(screen, (30, 30, 30), LIGHT_POS, MIN_SAFE_DISTANCE)
            
            # Draw Light
            pygame.draw.circle(screen, LIGHT_COLOR, LIGHT_POS, 40)
            
            # Draw Walls
            pygame.draw.rect(screen, WALL_COLOR, (0,0,WIDTH,WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0,HEIGHT-WALL_THICKNESS,WIDTH,WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0,0,WALL_THICKNESS,HEIGHT))
            pygame.draw.rect(screen, WALL_COLOR, (WIDTH-WALL_THICKNESS,0,WALL_THICKNESS,HEIGHT))

            # Draw Blocks (CIRCLES now)
            for b in blocks:
                pygame.draw.circle(screen, BLOCK_COLOR, (int(b.x), int(b.y)), int(b.radius))
                # Optional: Draw a small dot in center to see rotation/movement clearly
                pygame.draw.circle(screen, (60, 60, 180), (int(b.x), int(b.y)), 3)

            # Draw Robots
            for r in robots:
                color = REST_COLOR if r.in_shadow else ROBOT_COLOR
                pygame.draw.circle(screen, color, (int(r.x), int(r.y)), ROBOT_RADIUS)
                # Eyes
                ex = r.x + math.cos(r.angle) * (ROBOT_RADIUS-5)
                ey = r.y + math.sin(r.angle) * (ROBOT_RADIUS-5)
                pygame.draw.circle(screen, (255,255,255), (int(ex), int(ey)), 3)

            # Instructions
            draw_text(screen, "Press ENTER for Metrics", 16, 20, HEIGHT-30)
            pygame.display.flip()
            clock.tick(60)

        else:
            # Show Results
            dur = final_time if final_time else (time.time() - start_time)
            show_results(screen, blocks, dur)
            running = False

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()