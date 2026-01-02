import pygame   # The main library for graphics and window handling
import math     # Standard math functions
import random   # To generate random positions
import sys      # To interact with the system
import time     # To track simulation duration

# =============================================================================
# SECTION 1: GLOBAL CONFIGURATION
# =============================================================================

# -- SCENARIO SETTINGS --
# Scenario: "The Discovery"
# Blocks are everywhere, but the swarm enters from one door (Top-Left).
BLOCK_SPAWN_MODE = "UNIFORM"

# -- WINDOW SETTINGS --
WIDTH, HEIGHT = 600, 600
BG_COLOR = (20, 20, 20)
LIGHT_COLOR = (255, 255, 240)
LIGHT_POS = (WIDTH // 2, HEIGHT // 2)

# -- COLORS --
ROBOT_COLOR = (220, 60, 60)       # Red (Panic)
REST_COLOR = (60, 220, 60)        # Green (Rest)
BLOCK_COLOR = (80, 80, 220)       # Blue (Blocks)
WALL_COLOR = (255, 255, 0)        # Yellow (Walls)
TEXT_COLOR = (255, 255, 255)      # White

# -- ENTITIES --
NUM_ROBOTS = 20
NUM_BLOCKS = 25
BLOCK_SIZE = 30
ROBOT_RADIUS = 10
WALL_THICKNESS = 10

# -- LOGIC & PHYSICS --
MIN_SAFE_DISTANCE = 300           # Glare Zone radius
ROBOT_SPEED_PANIC = 3.5
ROBOT_SPEED_PUSH = 1.0
PUSH_FORCE = 1.0
BLOCK_FRICTION = 0.90

MAX_POSSIBLE_DIST = math.hypot(WIDTH // 2, HEIGHT // 2)

# =============================================================================
# SECTION 2: HELPER FUNCTIONS (SPAWNING LOGIC)
# =============================================================================

def get_block_spawn_pos(mode):
    """
    Determines where BLOCKS appear.
    """
    safe_min = WALL_THICKNESS + 20
    safe_max_x = WIDTH - WALL_THICKNESS - 20
    safe_max_y = HEIGHT - WALL_THICKNESS - 20
    
    # ALWAYS UNIFORM FOR THIS SCENARIO
    x = random.randint(safe_min, safe_max_x)
    y = random.randint(safe_min, safe_max_y)
    
    return x, y

def get_robot_spawn_pos():
    """
    Determines where ROBOTS appear.
    SCENARIO: All robots enter from the Top-Left Quadrant.
    """
    safe_min = WALL_THICKNESS + 20
    
    # We constrain the spawn area to the first half of the screen
    quadrant_limit_x = (WIDTH // 2) - 20
    quadrant_limit_y = (HEIGHT // 2) - 20
    
    x = random.randint(safe_min, quadrant_limit_x)
    y = random.randint(safe_min, quadrant_limit_y)
    
    return x, y

# =============================================================================
# SECTION 3: CLASS DEFINITIONS
# =============================================================================

class Robot:
    def __init__(self):
        # ROBOTS use the Quadrant spawner
        self.x, self.y = get_robot_spawn_pos()
        self.angle = random.uniform(0, 2 * math.pi)
        self.touching_block = False 
        self.in_shadow = False

    def sense(self, blocks, light_pos):
        self.in_shadow = False
        self.touching_block = False
        
        dx = self.x - light_pos[0]
        dy = self.y - light_pos[1]
        dist = math.hypot(dx, dy)
        
        if dist == 0: return 

        if dist < MIN_SAFE_DISTANCE:
            self.in_shadow = False 
            return 

        dir_x = -(dx / dist)
        dir_y = -(dy / dist)
        sensor_x = self.x + (dir_x * ROBOT_RADIUS)
        sensor_y = self.y + (dir_y * ROBOT_RADIUS)
        
        line_start = light_pos
        line_end = (sensor_x, sensor_y)
        
        for block in blocks:
            rect = pygame.Rect(int(block.x), int(block.y), block.w, block.h)
            if rect.clipline(line_start, line_end):
                self.in_shadow = True
                break

    def decide_and_move(self):
        if self.in_shadow:
            return 
        if self.touching_block:
            speed = ROBOT_SPEED_PUSH
        else:
            speed = ROBOT_SPEED_PANIC
            self.angle += random.uniform(-0.4, 0.4)

        self.x += math.cos(self.angle) * speed
        self.y += math.sin(self.angle) * speed
        self.force_bounds()

    def force_bounds(self):
        min_bound = WALL_THICKNESS + ROBOT_RADIUS
        max_x_bound = WIDTH - WALL_THICKNESS - ROBOT_RADIUS
        max_y_bound = HEIGHT - WALL_THICKNESS - ROBOT_RADIUS
        self.x = max(min_bound, min(max_x_bound, self.x))
        self.y = max(min_bound, min(max_y_bound, self.y))

class Block:
    def __init__(self):
        # BLOCKS use the Uniform spawner
        self.x, self.y = get_block_spawn_pos(BLOCK_SPAWN_MODE)
        
        # Data Metrics
        self.start_x = self.x
        self.start_y = self.y
        
        self.w = BLOCK_SIZE
        self.h = BLOCK_SIZE
        self.vx = 0.0
        self.vy = 0.0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= BLOCK_FRICTION
        self.vy *= BLOCK_FRICTION
        
        min_bound = WALL_THICKNESS
        max_x_bound = WIDTH - WALL_THICKNESS - self.w
        max_y_bound = HEIGHT - WALL_THICKNESS - self.h

        if self.x < min_bound: self.x = min_bound; self.vx = 0
        if self.x > max_x_bound: self.x = max_x_bound; self.vx = 0
        if self.y < min_bound: self.y = min_bound; self.vy = 0
        if self.y > max_y_bound: self.y = max_y_bound; self.vy = 0

# =============================================================================
# SECTION 4: PHYSICS & METRICS
# =============================================================================
def check_collisions(robots, blocks):
    # Robot vs Block
    for robot in robots:
        r_rect = pygame.Rect(int(robot.x - ROBOT_RADIUS), int(robot.y - ROBOT_RADIUS), 
                             ROBOT_RADIUS*2, ROBOT_RADIUS*2)
        for block in blocks:
            b_rect = pygame.Rect(int(block.x), int(block.y), block.w, block.h)
            if r_rect.colliderect(b_rect):
                robot.touching_block = True
                dx = (block.x + block.w/2) - robot.x
                dy = (block.y + block.h/2) - robot.y
                dist = math.hypot(dx, dy)
                if dist < 0.1: dist = 1.0 
                nx = dx/dist
                ny = dy/dist
                block.vx += nx * PUSH_FORCE
                block.vy += ny * PUSH_FORCE
                robot.x -= nx * 2
                robot.y -= ny * 2

    # Block vs Block
    for i, b1 in enumerate(blocks):
        for b2 in blocks[i+1:]:
            r1 = pygame.Rect(int(b1.x), int(b1.y), b1.w, b1.h)
            r2 = pygame.Rect(int(b2.x), int(b2.y), b2.w, b2.h)
            if r1.colliderect(r2):
                dx = b1.x - b2.x
                dy = b1.y - b2.y
                dist = math.hypot(dx, dy)
                if dist < 0.1: dist = 1.0
                push = 2.0
                b1.x += (dx/dist)*push; b1.y += (dy/dist)*push
                b2.x -= (dx/dist)*push; b2.y -= (dy/dist)*push

def draw_text(screen, text, size, x, y, color=TEXT_COLOR):
    font = pygame.font.SysFont("Arial", size, bold=False)
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))

def get_peripherality_score(x, y):
    dx = x - LIGHT_POS[0]
    dy = y - LIGHT_POS[1]
    dist = math.hypot(dx, dy)
    percent = (dist / MAX_POSSIBLE_DIST) * 100.0
    return percent

def show_results_screen(screen, blocks, duration):
    total_start_periph = 0
    total_end_periph = 0
    block_stats = []

    for i, b in enumerate(blocks):
        p_start = get_peripherality_score(b.start_x, b.start_y)
        p_end = get_peripherality_score(b.x, b.y)
        change = p_end - p_start
        total_start_periph += p_start
        total_end_periph += p_end
        block_stats.append(f"Block {i+1:02d}: {p_end:.1f}% ({change:+.1f}%)".replace('+', ''))

    avg_end = total_end_periph / len(blocks)
    avg_change = avg_end - (total_start_periph / len(blocks))

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False

        screen.fill(BG_COLOR)
        draw_text(screen, "SIMULATION COMPLETE", 36, 30, 30, (255, 215, 0))
        
        # Display Spawn Mode
        draw_text(screen, f"Scenario: Top-Left Quadrant Invasion", 20, 30, 70, (100, 200, 255))

        draw_text(screen, f"Avg Peripherality: {avg_end:.1f}%", 28, 30, 100)
        sign_str = "" if avg_change < 0 else ""
        draw_text(screen, f"Net Change:        {sign_str}{avg_change:.1f}%", 24, 30, 140, (200, 200, 200))
        draw_text(screen, f"Time to Finish:    {duration:.1f} sec", 24, 30, 180)
        
        if avg_change > 0:
            verdict = "VERDICT: SUCCESS (Expansion)"
            v_color = REST_COLOR
        else:
            verdict = "VERDICT: FAILURE (Contraction)"
            v_color = ROBOT_COLOR
        draw_text(screen, verdict, 24, 30, 220, v_color)

        draw_text(screen, "Individual Block Data:", 20, 30, 260, (150, 150, 150))
        
        y_start = 290
        line_height = 24
        for i in range(13):
            if i < len(block_stats):
                draw_text(screen, block_stats[i], 18, 30, y_start + (i * line_height))
        for i in range(13, len(block_stats)):
            draw_text(screen, block_stats[i], 18, 280, y_start + ((i-13) * line_height))

        draw_text(screen, "Press [Close Window] to exit.", 16, 30, HEIGHT - 30, (100, 100, 100))

        pygame.display.flip()
        pygame.time.Clock().tick(15)

# =============================================================================
# SECTION 5: MAIN EXECUTION
# =============================================================================
def main():
    print(f"Initializing Simulation (Scenario: Quadrant Spawn)...")
    pygame.init()
    pygame.font.init() 
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Type I Swarm: Quadrant Spawn")
    clock = pygame.time.Clock()

    # -- SPAWN SETUP --
    robots = [Robot() for _ in range(NUM_ROBOTS)]
    blocks = [Block() for _ in range(NUM_BLOCKS)]

    start_time = time.time() 
    final_time = None 
    
    running = True
    simulation_active = True 

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN: 
                    simulation_active = False 
                if event.key == pygame.K_SPACE:
                    main() 
                    return

        if simulation_active:
            check_collisions(robots, blocks)
            for b in blocks: b.update()
            
            safe_robots = 0
            for r in robots:
                r.sense(blocks, LIGHT_POS)
                r.decide_and_move()
                r.force_bounds()
                if r.in_shadow:
                    safe_robots += 1
            
            if safe_robots == NUM_ROBOTS and final_time is None:
                final_time = time.time() - start_time
            
            screen.fill(BG_COLOR)
            pygame.draw.circle(screen, (30, 30, 30), LIGHT_POS, MIN_SAFE_DISTANCE)
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, HEIGHT-WALL_THICKNESS, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WALL_THICKNESS, HEIGHT))
            pygame.draw.rect(screen, WALL_COLOR, (WIDTH-WALL_THICKNESS, 0, WALL_THICKNESS, HEIGHT))

            pygame.draw.circle(screen, LIGHT_COLOR, LIGHT_POS, 40)
            
            for b in blocks: 
                pygame.draw.rect(screen, BLOCK_COLOR, (int(b.x), int(b.y), b.w, b.h))
            
            for r in robots:
                color = REST_COLOR if r.in_shadow else ROBOT_COLOR
                pygame.draw.circle(screen, color, (int(r.x), int(r.y)), ROBOT_RADIUS)
                eye_x = r.x + math.cos(r.angle) * (ROBOT_RADIUS - 5)
                eye_y = r.y + math.sin(r.angle) * (ROBOT_RADIUS - 5)
                pygame.draw.circle(screen, (255, 255, 255), (int(eye_x), int(eye_y)), 3)

            draw_text(screen, f"Scenario: Top-Left Entry", 16, 20, 20, (150, 150, 150))
            draw_text(screen, "Press ENTER for Metrics", 16, 20, HEIGHT - 30)

            pygame.display.flip()
            clock.tick(60)
        
        else:
            if final_time is not None:
                duration_to_show = final_time
            else:
                duration_to_show = time.time() - start_time
            show_results_screen(screen, blocks, duration_to_show)
            running = False 

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()