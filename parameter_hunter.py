import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import math
import random
import numpy as np
from scipy.optimize import minimize

# =============================================================================
# OPTIMIZER CONFIGURATION
# =============================================================================
TRIALS_PER_EVALUATION = 5  # Number of trials to average per guess (higher = more stable, slower)
TRIAL_STEPS = 1000         # Evaluate at exactly 1000 steps

# Arena globals (needed for physics)
WIDTH, HEIGHT = 600, 600
NUM_BLOCKS = 20                   
BLOCK_RADIUS = 20                 
WALL_THICKNESS = 10               
GRID_SIZE = 10 
NUM_VENTS = GRID_SIZE * GRID_SIZE 
NUM_BBOTS = 50                    
BBOT_RADIUS = 3                   
BBOT_SPEED_BASE = 5               

# Variables we will be actively changing (default placeholders)
VENT_STRENGTH_WALL = -4.0
VENT_STRENGTH_SUB_WALL = -2.0
VENT_STRENGTH_INTERNAL = 0.5
VENT_REACH = 8
THERMAL_KICK = 1.0

# =============================================================================
# PHYSICS CLASSES (Stripped down for pure speed)
# =============================================================================
class Block:
    def __init__(self, x, y):
        self.pos = pygame.math.Vector2(x, y)
        self.start_pos = pygame.math.Vector2(x, y)

    def calculate_displacement(self):
        return self.pos.distance_to(self.start_pos)

class BBot:
    def __init__(self, x, y):
        self.pos = pygame.math.Vector2(x, y)
        self.velocity = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        if self.velocity.length() > 0:
            self.velocity.scale_to_length(BBOT_SPEED_BASE)
        self.in_zone = False

# =============================================================================
# CORE SIMULATION LOGIC
# =============================================================================
def apply_vent_logic(bot):
    grid_w = (WIDTH - 2 * WALL_THICKNESS) / GRID_SIZE
    grid_h = (HEIGHT - 2 * WALL_THICKNESS) / GRID_SIZE
    
    col = int((bot.pos.x - WALL_THICKNESS) // grid_w)
    row = int((bot.pos.y - WALL_THICKNESS) // grid_h)
    
    col = max(0, min(GRID_SIZE - 1, col))
    row = max(0, min(GRID_SIZE - 1, row))
    
    dist_to_wall = min(col, row, GRID_SIZE - 1 - col, GRID_SIZE - 1 - row)
    
    if dist_to_wall == 0:
        strength = VENT_STRENGTH_WALL
    elif dist_to_wall == 1:
        strength = VENT_STRENGTH_SUB_WALL
    else:
        strength = VENT_STRENGTH_INTERNAL

    kick_x = random.uniform(-THERMAL_KICK, THERMAL_KICK)
    kick_y = random.uniform(-THERMAL_KICK, THERMAL_KICK)
    
    bot.velocity.x += strength + kick_x
    bot.velocity.y += strength + kick_y
    
    if bot.velocity.length() > 0:
        bot.velocity.scale_to_length(BBOT_SPEED_BASE)

def run_headless_trials():
    """Runs N trials and returns the average rock displacement."""
    total_mean_displacement = 0.0
    
    for _ in range(TRIALS_PER_EVALUATION):
        # Initialize Rocks
        blocks = []
        for _ in range(NUM_BLOCKS):
            placed = False
            while not placed:
                bx = random.uniform(WALL_THICKNESS + BLOCK_RADIUS, 150)
                by = random.uniform(WALL_THICKNESS + BLOCK_RADIUS, 150)
                new_pos = pygame.math.Vector2(bx, by)
                overlap = any(new_pos.distance_to(b.pos) < (BLOCK_RADIUS * 2) for b in blocks)
                if not overlap:
                    blocks.append(Block(bx, by))
                    placed = True

        # Initialize BBots
        bbots = []
        for _ in range(NUM_BBOTS):
            bx = random.uniform(WALL_THICKNESS + BBOT_RADIUS, 150)
            by = random.uniform(WALL_THICKNESS + BBOT_RADIUS, 150)
            bbots.append(BBot(bx, by))

        # Run Physics Loop
        for step in range(TRIAL_STEPS):
            for bot in bbots:
                apply_vent_logic(bot)
                bot.pos += bot.velocity

                # Bounce off walls
                if bot.pos.x - BBOT_RADIUS < WALL_THICKNESS:
                    bot.pos.x = WALL_THICKNESS + BBOT_RADIUS
                    bot.velocity.x *= -1
                elif bot.pos.x + BBOT_RADIUS > WIDTH - WALL_THICKNESS:
                    bot.pos.x = WIDTH - WALL_THICKNESS - BBOT_RADIUS
                    bot.velocity.x *= -1
                if bot.pos.y - BBOT_RADIUS < WALL_THICKNESS:
                    bot.pos.y = WALL_THICKNESS + BBOT_RADIUS
                    bot.velocity.y *= -1
                elif bot.pos.y + BBOT_RADIUS > HEIGHT - WALL_THICKNESS:
                    bot.pos.y = HEIGHT - WALL_THICKNESS - BBOT_RADIUS
                    bot.velocity.y *= -1

                # Bounce off Rocks (Osmotic force application)
                for b in blocks:
                    dist = bot.pos.distance_to(b.pos)
                    min_dist = BBOT_RADIUS + BLOCK_RADIUS
                    if dist < min_dist:
                        overlap = min_dist - dist
                        if dist > 0:
                            push_dir = (bot.pos - b.pos).normalize()
                            b.pos -= push_dir * overlap * 0.1 
                            bot.pos += push_dir * overlap * 0.9
                            bot.velocity.reflect_ip(push_dir)

            # Keep rocks in bounds
            for b in blocks:
                b.pos.x = max(WALL_THICKNESS + BLOCK_RADIUS, min(WIDTH - WALL_THICKNESS - BLOCK_RADIUS, b.pos.x))
                b.pos.y = max(WALL_THICKNESS + BLOCK_RADIUS, min(HEIGHT - WALL_THICKNESS - BLOCK_RADIUS, b.pos.y))

        # Calculate average displacement for this trial
        trial_disp = sum(b.calculate_displacement() for b in blocks) / NUM_BLOCKS
        total_mean_displacement += trial_disp

    return total_mean_displacement / TRIALS_PER_EVALUATION

# =============================================================================
# OPTIMIZATION COST FUNCTION
# =============================================================================
evaluation_count = 0

def cost_function(params, target):
    global VENT_STRENGTH_WALL, VENT_STRENGTH_SUB_WALL, VENT_STRENGTH_INTERNAL, VENT_REACH, THERMAL_KICK, evaluation_count
    
    # Update globals with current optimizer guess
    VENT_STRENGTH_WALL = params[0]
    VENT_STRENGTH_SUB_WALL = params[1]
    VENT_STRENGTH_INTERNAL = params[2]
    VENT_REACH = int(params[3])  # Reach must be an integer for grid logic
    THERMAL_KICK = params[4]

    # Run the simulation
    mean_disp = run_headless_trials()
    
    # Calculate the error (cost)
    cost = (mean_disp - target)**2
    
    evaluation_count += 1
    print(f"Eval {evaluation_count:03d} | Params: [W:{params[0]:.1f}, SW:{params[1]:.1f}, I:{params[2]:.1f}, R:{params[3]:.1f}, K:{params[4]:.2f}] | Yields: {mean_disp:.2f}px | Diff: {(mean_disp - target):.2f}")
    
    return cost

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    pygame.init()
    
    target_val = float(input("Enter target mean displacement at 1000 steps (e.g., 7.8): "))
    print("\nStarting optimization. This will run dozens of simulations. Please wait...\n")

    # Initial guess: [W_Wall, W_SubWall, W_Internal, Reach, Kick]
    initial_guess = [-4.0, -2.0, 0.5, 8.0, 1.0]

    # Run Nelder-Mead optimization
    result = minimize(
        cost_function, 
        initial_guess, 
        args=(target_val,), 
        method='Nelder-Mead',
        options={'maxiter': 100, 'xatol': 0.1, 'fatol': 0.1}
    )

    print("\n==================================================")
    if result.success or result.status == 2:  # Status 2 is maxiter reached, which is fine for our heuristic
        opt = result.x
        print("OPTIMIZATION COMPLETE!")
        print(f"Target Displacement: {target_val} px")
        print("--------------------------------------------------")
        print(f"VENT_STRENGTH_WALL     = {opt[0]:.4f}")
        print(f"VENT_STRENGTH_SUB_WALL = {opt[1]:.4f}")
        print(f"VENT_STRENGTH_INTERNAL = {opt[2]:.4f}")
        print(f"VENT_REACH             = {int(opt[3])}")
        print(f"THERMAL_KICK           = {opt[4]:.4f}")
    else:
        print("Optimizer failed to converge.")
        print(result.message)
    print("==================================================")