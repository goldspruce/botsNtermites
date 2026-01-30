import pygame
import math
import random
import sys
import copy
import time

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# -- WINDOW & WORLD --
WIDTH, HEIGHT = 800, 600
BG_COLOR = (20, 20, 20)
TEXT_COLOR = (255, 255, 255)
LIGHT_POS = (WIDTH // 2, HEIGHT // 2)

# -- EVOLUTION SETTINGS --
POPULATION_SIZE = 15        # How many different "Species" to test per generation
GENERATIONS = 10            # How many generations to evolve
SIM_DURATION_TICKS = 1000   # How long a headless simulation lasts (frames)
MUTATION_RATE = 0.15        # Chance to change stats

# -- BASE PHYSICS CONSTANTS --
BLOCK_SIZE = 30
NUM_BLOCKS = 25
NUM_ROBOTS = 20
WALL_THICKNESS = 10
MIN_SAFE_DISTANCE = 300     # Radius of the light glare
BLOCK_FRICTION = 0.85

# =============================================================================
# 2. GENETICS (SWARM DNA)
# =============================================================================

class SwarmGenome:
    def __init__(self, speed=None, radius=None, force=None):
        # If no genes provided, randomize (Generation 0)
        self.speed = speed if speed else random.uniform(2.0, 8.0)
        self.radius = radius if radius else random.uniform(5.0, 15.0)
        self.force = force if force else random.uniform(0.5, 5.0)
        
        # Fitness tracking
        self.fitness = 0.0

    def __repr__(self):
        return f"[Spd:{self.speed:.1f} | Rad:{self.radius:.1f} | Frc:{self.force:.1f}]"

def mutate(genome):
    """ Returns a mutated copy of the genome """
    new_g = copy.copy(genome)
    
    if random.random() < MUTATION_RATE:
        new_g.speed += random.uniform(-1.0, 1.0)
    if random.random() < MUTATION_RATE:
        new_g.radius += random.uniform(-2.0, 2.0)
    if random.random() < MUTATION_RATE:
        new_g.force += random.uniform(-0.5, 0.5)
    
    # Clamp values to logical limits
    new_g.speed = max(1.0, min(10.0, new_g.speed))
    new_g.radius = max(4.0, min(20.0, new_g.radius))
    new_g.force = max(0.1, min(8.0, new_g.force))
    
    new_g.fitness = 0.0 # Reset fitness for new entity
    return new_g

def crossover(parent_a, parent_b):
    """ Mixes traits of two successful swarms """
    # Randomly pick genes from either parent
    s = parent_a.speed if random.random() > 0.5 else parent_b.speed
    r = parent_a.radius if random.random() > 0.5 else parent_b.radius
    f = parent_a.force if random.random() > 0.5 else parent_b.force
    return SwarmGenome(s, r, f)

# =============================================================================
# 3. ENTITIES
# =============================================================================

class Robot:
    def __init__(self, genome):
        # All robots in this run share the SAME genome
        self.radius = genome.radius
        self.speed = genome.speed
        self.force = genome.force
        
        # Spawn logic
        safe_min = WALL_THICKNESS + 20
        limit_x = (WIDTH // 2) - 20
        limit_y = (HEIGHT // 2) - 20
        self.x = random.randint(safe_min, limit_x)
        self.y = random.randint(safe_min, limit_y)
        self.angle = random.uniform(0, 2 * math.pi)
        
        self.touching_block = False
        self.in_shadow = False

    def sense_and_act(self, blocks, light_pos):
        # 1. Check Light
        dx = self.x - light_pos[0]
        dy = self.y - light_pos[1]
        dist = math.hypot(dx, dy)
        
        self.in_shadow = False
        
        if dist > MIN_SAFE_DISTANCE:
            # Check for shadows only if inside glare zone logic isn't saving us
            # Actually, let's say inside glare zone = danger.
            # Outside glare zone (if we defined it) = safe? 
            # Let's stick to: Must be behind block to be safe if close.
            pass

        if dist < MIN_SAFE_DISTANCE:
            # Raycast
            if dist == 0: return
            dir_x, dir_y = -(dx/dist), -(dy/dist)
            sensor_x = self.x + (dir_x * self.radius)
            sensor_y = self.y + (dir_y * self.radius)
            line_end = (sensor_x, sensor_y)
            
            # Simple line check
            for b in blocks:
                # Optimized rect collision for ray
                if b.rect.clipline(light_pos, line_end):
                    self.in_shadow = True
                    break
        
        # 2. Move
        if self.in_shadow:
            return # Stay put
        
        # Move behavior
        move_speed = self.speed
        if self.touching_block:
            move_speed = self.speed * 0.5 # Pushing is slower
        else:
            self.angle += random.uniform(-0.5, 0.5) # Jitter

        self.x += math.cos(self.angle) * move_speed
        self.y += math.sin(self.angle) * move_speed
        
        # Bounds
        m = WALL_THICKNESS + self.radius
        self.x = max(m, min(WIDTH - m, self.x))
        self.y = max(m, min(HEIGHT - m, self.y))

class Block:
    def __init__(self):
        self.w = BLOCK_SIZE
        self.h = BLOCK_SIZE
        # Uniform random spawn
        self.x = random.randint(50, WIDTH - 50)
        self.y = random.randint(50, HEIGHT - 50)
        self.vx, self.vy = 0.0, 0.0
        self.rect = pygame.Rect(self.x, self.y, self.w, self.h)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= BLOCK_FRICTION
        self.vy *= BLOCK_FRICTION
        
        # Bounds
        mx = WIDTH - WALL_THICKNESS - self.w
        my = HEIGHT - WALL_THICKNESS - self.h
        self.x = max(WALL_THICKNESS, min(mx, self.x))
        self.y = max(WALL_THICKNESS, min(my, self.y))
        
        self.rect.x = int(self.x)
        self.rect.y = int(self.y)

# =============================================================================
# 4. PHYSICS ENGINE (SEPARATED)
# =============================================================================

def run_physics_step(robots, blocks):
    # Robot vs Block
    for r in robots:
        r_rect = pygame.Rect(int(r.x - r.radius), int(r.y - r.radius), int(r.radius*2), int(r.radius*2))
        r.touching_block = False
        
        for b in blocks:
            if r_rect.colliderect(b.rect):
                r.touching_block = True
                
                # Push math
                dx = (b.x + b.w/2) - r.x
                dy = (b.y + b.h/2) - r.y
                dist = math.hypot(dx, dy)
                if dist == 0: dist = 0.1
                
                nx, ny = dx/dist, dy/dist
                
                # FORCE comes from Genome
                b.vx += nx * r.force
                b.vy += ny * r.force
                
                # Robot bounce back
                r.x -= nx * (r.speed * 0.5)
                r.y -= ny * (r.speed * 0.5)

    # Block vs Block
    for i, b1 in enumerate(blocks):
        for b2 in blocks[i+1:]:
            if b1.rect.colliderect(b2.rect):
                dx = b1.x - b2.x
                dy = b1.y - b2.y
                dist = math.hypot(dx, dy)
                if dist == 0: dist = 1.0
                push = 1.5
                nx, ny = dx/dist, dy/dist
                
                b1.x += nx * push
                b1.y += ny * push
                b2.x -= nx * push
                b2.y -= ny * push

# =============================================================================
# 5. SIMULATION RUNNERS
# =============================================================================

def run_headless_simulation(genome):
    """ 
    Runs the simulation mathematically without drawing. 
    Returns: Fitness Score (Total frames spent in shadow by all robots)
    """
    # 1. Setup Environment
    robots = [Robot(genome) for _ in range(NUM_ROBOTS)]
    blocks = [Block() for _ in range(NUM_BLOCKS)]
    
    total_score = 0
    
    # 2. Run Loop
    for _ in range(SIM_DURATION_TICKS):
        # Physics
        run_physics_step(robots, blocks)
        for b in blocks: b.update()
        
        # Logic & Scoring
        robots_safe_this_frame = 0
        for r in robots:
            r.sense_and_act(blocks, LIGHT_POS)
            if r.in_shadow:
                robots_safe_this_frame += 1
        
        total_score += robots_safe_this_frame

    return total_score

def draw_visual_simulation(screen, genome, best_gen, best_score):
    """
    Runs the simulation WITH graphics for the user to watch.
    """
    clock = pygame.time.Clock()
    robots = [Robot(genome) for _ in range(NUM_ROBOTS)]
    blocks = [Block() for _ in range(NUM_BLOCKS)]
    
    running = True
    frame_count = 0
    
    font = pygame.font.SysFont("Courier New", 18)
    
    while running:
        # Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = False # Stop replay, go to next generation
        
        # Update
        run_physics_step(robots, blocks)
        for b in blocks: b.update()
        
        safe_count = 0
        for r in robots:
            r.sense_and_act(blocks, LIGHT_POS)
            if r.in_shadow: safe_count += 1
        
        # Draw
        screen.fill(BG_COLOR)
        
        # Draw Light
        pygame.draw.circle(screen, (30,30,30), LIGHT_POS, MIN_SAFE_DISTANCE)
        pygame.draw.circle(screen, (255,255,220), LIGHT_POS, 40)
        
        # Draw Blocks
        for b in blocks:
            pygame.draw.rect(screen, (80, 80, 200), b.rect)
            
        # Draw Robots
        for r in robots:
            # Color indicates state
            c = (50, 200, 50) if r.in_shadow else (200, 50, 50)
            pygame.draw.circle(screen, c, (int(r.x), int(r.y)), int(r.radius))
            # Draw Outline for visibility
            pygame.draw.circle(screen, (255,255,255), (int(r.x), int(r.y)), int(r.radius), 1)

        # UI Overlay
        # Box for stats
        pygame.draw.rect(screen, (0,0,0), (0, 0, 350, 160))
        pygame.draw.rect(screen, (255,255,255), (0, 0, 350, 160), 2)
        
        def draw_txt(txt, y):
            img = font.render(txt, True, TEXT_COLOR)
            screen.blit(img, (10, y))
            
        draw_txt(f"--- REPLAYING BEST SWARM ---", 10)
        draw_txt(f"Generation: {best_gen}", 35)
        draw_txt(f"Fitness Score: {best_score}", 55)
        draw_txt(f"Genome Traits:", 85)
        draw_txt(f" Speed:  {genome.speed:.2f}", 105)
        draw_txt(f" Radius: {genome.radius:.2f}", 125)
        draw_txt(f" Force:  {genome.force:.2f}", 145)
        
        draw_txt("Press [SPACE] to Evolve Next Gen", HEIGHT - 30)

        pygame.display.flip()
        clock.tick(60)
        frame_count += 1
        
        if frame_count > 1000: # Auto-end replay after some time
             running = False

# =============================================================================
# 6. MAIN CONTROLLER
# =============================================================================

def draw_loading_screen(screen, current_gen, species_idx, best_so_far):
    screen.fill(BG_COLOR)
    font = pygame.font.SysFont("Courier New", 20)
    
    lines = [
        f"EVOLVING GENERATION {current_gen} / {GENERATIONS}",
        f"-----------------------------------",
        f"Simulating Species: {species_idx} / {POPULATION_SIZE}",
        f"",
        f"Current Best Fitness: {best_so_far:.0f}",
        f"",
        f"[ Calculating Physics in Background... ]"
    ]
    
    y = HEIGHT // 3
    for line in lines:
        s = font.render(line, True, (0, 255, 0))
        screen.blit(s, (WIDTH//2 - s.get_width()//2, y))
        y += 30
        
    pygame.display.flip()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Swarm Evolution Lab")
    
    # 1. Initialize Population
    population = [SwarmGenome() for _ in range(POPULATION_SIZE)]
    
    global_best_genome = None
    global_best_score = 0
    
    for generation in range(1, GENERATIONS + 1):
        
        # --- PHASE 1: HEADLESS TRAINING ---
        # Evaluate every species in the population
        for i, genome in enumerate(population):
            
            # Show progress on screen (because headless takes a second)
            draw_loading_screen(screen, generation, i+1, global_best_score)
            
            # Run simulation (CPU only, no graphics)
            fitness = run_headless_simulation(genome)
            genome.fitness = fitness
            
            # Check for events so window doesn't freeze
            pygame.event.pump() 

        # --- PHASE 2: SELECTION ---
        # Sort by fitness (High is better)
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        current_best = population[0]
        if current_best.fitness > global_best_score:
            global_best_score = current_best.fitness
            global_best_genome = copy.deepcopy(current_best)
            
        print(f"GEN {generation} COMPLETE. Best Fitness: {current_best.fitness}")
        print(f"Best Genome: {current_best}")

        # --- PHASE 3: VISUAL REPLAY ---
        # Show the user what the best swarm looks like
        draw_visual_simulation(screen, current_best, generation, current_best.fitness)
        
        # --- PHASE 4: REPRODUCTION ---
        # Elitism: Keep top 20%
        num_elites = int(POPULATION_SIZE * 0.2)
        next_gen = population[:num_elites]
        
        # Fill rest with children
        while len(next_gen) < POPULATION_SIZE:
            parent_a = random.choice(population[:num_elites*2]) # Bias to top
            parent_b = random.choice(population[:num_elites*2])
            
            child = crossover(parent_a, parent_b)
            child = mutate(child)
            next_gen.append(child)
            
        population = next_gen

    pygame.quit()

if __name__ == "__main__":
    main()