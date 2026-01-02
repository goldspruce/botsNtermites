import pygame
import math
import random
import sys
import copy

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
WIDTH, HEIGHT = 800, 600
PLAY_WIDTH = 600
BG_COLOR = (20, 20, 20)
UI_BG_COLOR = (40, 40, 40)
TEXT_COLOR = (255, 255, 255)

# -- EVOLUTION SETTINGS --
POPULATION_SIZE = 12       # Smaller pop for faster generations
GENERATIONS = 20           # Convergence should happen quickly
TRIAL_STEPS = 600          # 10 seconds per simulation (Headless is fast)
MUTATION_RATE = 0.4
MUTATION_AMOUNT = 0.1

# -- SCENARIO CONSTANTS --
NUM_ROBOTS = 20
NUM_BLOCKS = 30            # Slightly more blocks to make it harder
BLOCK_SIZE = 30
WALL_THICKNESS = 10
LIGHT_POS = (PLAY_WIDTH // 2, HEIGHT // 2)

# -- CONSTANT PHYSICS (NOT EVOLVED) --
ROBOT_RADIUS = 10
PUSH_FORCE = 1.0           # Fixed at 1.0
BLOCK_FRICTION = 0.85      # High friction so blocks stop quickly

# =============================================================================
# 2. GENETICS ENGINE (SINGLE PARAMETER)
# =============================================================================
class Genome:
    def __init__(self, speed_gene=None):
        # GENE: Speed (0.0 to 1.0)
        # Maps to actual speeds of 1.0 to 8.0
        self.gene = speed_gene if speed_gene is not None else random.random()
        self.fitness = 0.0

    def get_speed(self):
        # Map 0.0-1.0 to 1.5-8.5 pixels per frame
        return 1.5 + (self.gene * 7.0)

    def mutate(self):
        new_val = self.gene
        if random.random() < MUTATION_RATE:
            change = random.uniform(-MUTATION_AMOUNT, MUTATION_AMOUNT)
            new_val = max(0.0, min(1.0, new_val + change))
        return Genome(new_val)

# =============================================================================
# 3. SIMULATION ENTITIES
# =============================================================================
class Robot:
    def __init__(self, speed_px):
        # SPAWN: Top-Left Quadrant
        safe = WALL_THICKNESS + ROBOT_RADIUS + 5
        self.x = random.randint(safe, (PLAY_WIDTH//2)-20)
        self.y = random.randint(safe, (HEIGHT//2)-20)
        self.angle = random.uniform(0, 6.28)
        
        # DNA Trait
        self.speed = speed_px
        
        self.in_shadow = False

    def update(self, blocks):
        self.in_shadow = False
        
        # 1. Light Sensing (Raycast)
        dx, dy = self.x - LIGHT_POS[0], self.y - LIGHT_POS[1]
        dist = math.hypot(dx, dy)
        
        # Only check shadow if we are close enough (Glare Zone)
        if dist < 300: 
            dir_x, dir_y = -(dx/dist), -(dy/dist)
            sensor_x = self.x + dir_x * ROBOT_RADIUS
            sensor_y = self.y + dir_y * ROBOT_RADIUS
            
            line_end = (sensor_x, sensor_y)
            for b in blocks:
                if b.rect.clipline(LIGHT_POS, line_end):
                    self.in_shadow = True
                    break
        
        # 2. Movement State
        if self.in_shadow: 
            return # STOP if safe
        
        # Move forward
        # Small wiggle (Brownian)
        self.angle += random.uniform(-0.2, 0.2)
            
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        
        # 3. Wall Bounds
        m = WALL_THICKNESS + ROBOT_RADIUS
        self.x = max(m, min(PLAY_WIDTH-m, self.x))
        self.y = max(m, min(HEIGHT-m, self.y))

class Block:
    def __init__(self):
        safe = WALL_THICKNESS + 20
        self.w, self.h = BLOCK_SIZE, BLOCK_SIZE
        # Random spawn
        self.x = random.randint(safe, PLAY_WIDTH - safe - self.w)
        self.y = random.randint(safe, HEIGHT - safe - self.h)
        self.vx, self.vy = 0, 0
        self.rect = pygame.Rect(self.x, self.y, self.w, self.h)
        
        self.start_dist = self.get_dist_from_light()

    def get_dist_from_light(self):
        cx, cy = self.x + self.w/2, self.y + self.h/2
        return math.hypot(cx - LIGHT_POS[0], cy - LIGHT_POS[1])

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= BLOCK_FRICTION
        self.vy *= BLOCK_FRICTION
        
        m = WALL_THICKNESS
        self.x = max(m, min(PLAY_WIDTH - m - self.w, self.x))
        self.y = max(m, min(HEIGHT - m - self.h, self.y))
        self.rect.topleft = (self.x, self.y)

# =============================================================================
# 4. PHYSICS ENGINE
# =============================================================================
def resolve_physics(robots, blocks):
    # Robot -> Block
    for r in robots:
        # Simple Circle-Rect collision
        test_x = max(min(r.x, 2000), 0) # dummy clamp
        
        # Find closest point on rect to circle
        for b in blocks:
            closest_x = max(b.x, min(r.x, b.x + b.w))
            closest_y = max(b.y, min(r.y, b.y + b.h))
            
            dx = r.x - closest_x
            dy = r.y - closest_y
            dist_sq = dx*dx + dy*dy
            
            if dist_sq < (ROBOT_RADIUS * ROBOT_RADIUS):
                # Collision
                dist = math.sqrt(dist_sq)
                if dist == 0: dist = 0.1
                nx, ny = dx/dist, dy/dist
                
                # Transfer Momentum
                # Force is constant, but frequency of hits depends on speed
                b.vx -= nx * PUSH_FORCE 
                b.vy -= ny * PUSH_FORCE
                
                # Bounce Robot
                r.x += nx * 2
                r.y += ny * 2

    # Block -> Block
    for i, b1 in enumerate(blocks):
        for b2 in blocks[i+1:]:
            if b1.rect.colliderect(b2.rect):
                dx = b1.rect.centerx - b2.rect.centerx
                dy = b1.rect.centery - b2.rect.centery
                dist = math.hypot(dx, dy)
                if dist == 0: dist = 1
                
                push = 1.0
                b1.x += (dx/dist)*push; b1.y += (dy/dist)*push
                b2.x -= (dx/dist)*push; b2.y -= (dy/dist)*push

# =============================================================================
# 5. SIMULATION RUNNER
# =============================================================================
def run_generation(genome, visual=False, screen=None, font=None):
    robots = [Robot(genome.get_speed()) for _ in range(NUM_ROBOTS)]
    blocks = [Block() for _ in range(NUM_BLOCKS)]
    
    clock = pygame.time.Clock() if visual else None
    
    for step in range(TRIAL_STEPS):
        resolve_physics(robots, blocks)
        for b in blocks: b.update()
        for r in robots: r.update(blocks)
        
        if visual:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            
            screen.fill(BG_COLOR)
            pygame.draw.rect(screen, UI_BG_COLOR, (PLAY_WIDTH, 0, WIDTH-PLAY_WIDTH, HEIGHT))
            
            # Arena
            pygame.draw.circle(screen, (30,30,30), LIGHT_POS, 300)
            pygame.draw.circle(screen, (255,255,220), LIGHT_POS, 40)
            
            for b in blocks:
                pygame.draw.rect(screen, (80, 80, 200), b.rect)
                
            for r in robots:
                col = (60, 220, 60) if r.in_shadow else (220, 60, 60)
                pygame.draw.circle(screen, col, (int(r.x), int(r.y)), ROBOT_RADIUS)
                
            # Stats
            lbls = [
                f"REPLAYING BEST",
                f"Generation Speed:",
                f"{genome.get_speed():.2f} px/frame",
                f"",
                f"Gene: {genome.gene:.2f}",
            ]
            for i, s in enumerate(lbls):
                t = font.render(s, True, TEXT_COLOR)
                screen.blit(t, (PLAY_WIDTH+20, 50 + i*25))

            pygame.display.flip()
            clock.tick(60)

    # Fitness = Average increase in block distance from light
    # We want blocks pushed to the edges (high peripherality)
    start_avg = sum(b.start_dist for b in blocks) / len(blocks)
    end_avg = sum(b.get_dist_from_light() for b in blocks) / len(blocks)
    
    return end_avg - start_avg

# =============================================================================
# 6. MAIN LOOP
# =============================================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Evolution: Finding Optimal Panic Speed")
    font = pygame.font.SysFont("monospace", 16, bold=True)
    
    population = [Genome() for _ in range(POPULATION_SIZE)]
    current_gen = 0
    state = "EVOLVING"
    best_genome = None
    
    history_avg_fitness = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        if state == "EVOLVING":
            screen.fill(UI_BG_COLOR)
            t = font.render(f"Gen {current_gen+1}/{GENERATIONS}: Simulating {POPULATION_SIZE} timelines...", True, (0, 255, 0))
            screen.blit(t, (50, HEIGHT//2 - 50))
            pygame.display.flip()
            
            # Run simulation for every genome
            gen_scores = []
            for g in population:
                g.fitness = run_generation(g, visual=False)
                gen_scores.append(g)
            
            # Selection
            gen_scores.sort(key=lambda x: x.fitness, reverse=True)
            best_genome = gen_scores[0]
            
            avg_fit = sum(g.fitness for g in population) / len(population)
            history_avg_fitness.append(avg_fit)
            
            # Reproduction
            new_pop = [best_genome] # Elitism
            top_half = gen_scores[:len(gen_scores)//2]
            
            while len(new_pop) < POPULATION_SIZE:
                parent = random.choice(top_half)
                new_pop.append(parent.mutate())
            
            population = new_pop
            current_gen += 1
            state = "REPLAY"

        elif state == "REPLAY":
            # Show the best result of this generation
            run_generation(best_genome, visual=True, screen=screen, font=font)
            
            if current_gen < GENERATIONS:
                state = "EVOLVING"
            else:
                state = "DONE"

        elif state == "DONE":
            screen.fill(BG_COLOR)
            lines = [
                "SIMULATION FINISHED",
                f"Best Speed Found: {best_genome.get_speed():.2f} px/frame",
                f"Best Gene Val: {best_genome.gene:.2f}",
                "",
                "History (Fitness):"
            ]
            for i, l in enumerate(lines):
                t = font.render(l, True, (0,255,0))
                screen.blit(t, (WIDTH//2 - 150, 100 + i*30))
                
            # Draw tiny graph
            for i, val in enumerate(history_avg_fitness):
                h = val * 5 # scale
                pygame.draw.rect(screen, (255, 0, 0), (100 + i*20, 500, 15, -h))
            
            pygame.display.flip()

if __name__ == "__main__":
    main()