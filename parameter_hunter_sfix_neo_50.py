import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import math
import random
import numpy as np
import concurrent.futures
from scipy.optimize import minimize

# =============================================================================
# OPTIMIZER CONFIGURATION
# =============================================================================
TRIALS_PER_EVALUATION = 50  # Number of parallel trials (Maxes out your Mac's performance cores)
TRIAL_STEPS = 700          # Default; will be dynamically overridden by user input

# Arena globals (needed for physics)
WIDTH, HEIGHT = 600, 600
WALL_THICKNESS = 10
LIGHT_POS = (15, 15)

NUM_BLOCKS = 20
BLOCK_RADIUS = 20
GRID_SIZE = 10
VENT_REACH = 60
BLOCK_FRICTION = 0.90
RESTITUTION = 0.5

# =============================================================================
# PHYSICS CLASSES & HELPERS
# =============================================================================
def get_valid_block_pos(existing_blocks):
    safe_min = WALL_THICKNESS + BLOCK_RADIUS
    safe_max_x = WIDTH - WALL_THICKNESS - BLOCK_RADIUS
    safe_max_y = HEIGHT - WALL_THICKNESS - BLOCK_RADIUS
    
    while True: 
        x = random.randint(int(safe_min), int(safe_max_x))
        y = random.randint(int(safe_min), int(safe_max_y))
        
        if any(math.hypot(x - blk.x, y - blk.y) < (BLOCK_RADIUS * 2 + 2) for blk in existing_blocks):
            continue
        return x, y

def get_distance_pixels(x, y):
    """Calculates radial distance from the Top-Left corner (15, 15)"""
    return math.hypot(x - LIGHT_POS[0], y - LIGHT_POS[1])

class Vent:
    def __init__(self, gx, gy):
        self.gx = gx
        self.gy = gy
        self.x = (gx * (WIDTH / GRID_SIZE)) + (WIDTH / (2 * GRID_SIZE))
        self.y = (gy * (HEIGHT / GRID_SIZE)) + (HEIGHT / (2 * GRID_SIZE))
        
        # Outer Ring (Touches the walls)
        self.is_boundary = (gx == 0 or gx == GRID_SIZE-1 or gy == 0 or gy == GRID_SIZE-1)
        # Second Ring (One removed from the walls)
        self.is_sub_boundary = not self.is_boundary and (gx == 1 or gx == GRID_SIZE-2 or gy == 1 or gy == GRID_SIZE-2)
        
        self.vx, self.vy = 0.0, 0.0

    def update_wind(self, vw, vsw, vi):
        if self.is_boundary or self.is_sub_boundary:
            nx, ny = 0, 0
            if self.gx <= 1: nx = -1                
            elif self.gx >= GRID_SIZE - 2: nx = 1   
            if self.gy <= 1: ny = -1                
            elif self.gy >= GRID_SIZE - 2: ny = 1   
            
            dist = math.hypot(nx, ny)
            strength = vw if self.is_boundary else vsw

            if dist > 0:
                if dist > 1.0: 
                    nx = -nx 
                    ny = -ny
                else:
                    nx /= dist
                    ny /= dist
                    
            self.vx = nx * strength
            self.vy = ny * strength
        else:
            angle = random.uniform(0, 2 * math.pi)
            self.vx = math.cos(angle) * vi
            self.vy = math.sin(angle) * vi

class Block:
    def __init__(self, existing_blocks):
        self.radius = BLOCK_RADIUS
        self.x, self.y = get_valid_block_pos(existing_blocks)
        self.start_x, self.start_y = self.x, self.y 
        self.vx, self.vy = 0.0, 0.0 

    def apply_vent_physics(self, vents, tk):
        self.vx += random.uniform(-tk, tk)
        self.vy += random.uniform(-tk, tk)

        for v in vents:
            dist = math.hypot(self.x - v.x, self.y - v.y)
            if dist < VENT_REACH:
                power = 1.0 - (dist / VENT_REACH)
                self.vx += v.vx * power
                self.vy += v.vy * power

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= BLOCK_FRICTION
        self.vy *= BLOCK_FRICTION
        
        min_bound = WALL_THICKNESS + self.radius
        max_x_bound = WIDTH - min_bound
        max_y_bound = HEIGHT - min_bound

        if self.x <= min_bound: 
            self.x = min_bound
            self.vx = -self.vx * RESTITUTION
        elif self.x >= max_x_bound: 
            self.x = max_x_bound
            self.vx = -self.vx * RESTITUTION
            
        if self.y <= min_bound: 
            self.y = min_bound
            self.vy = -self.vy * RESTITUTION
        elif self.y >= max_y_bound: 
            self.y = max_y_bound
            self.vy = -self.vy * RESTITUTION

def check_collisions(blocks):
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            b1 = blocks[i]
            b2 = blocks[j]
            dx = b1.x - b2.x
            dy = b1.y - b2.y
            dist = math.hypot(dx, dy)
            min_dist = b1.radius + b2.radius
            
            if dist < min_dist and dist > 0:
                overlap = min_dist - dist
                nx = dx / dist
                ny = dy / dist
                
                b1.x += nx * (overlap / 2)
                b1.y += ny * (overlap / 2)
                b2.x -= nx * (overlap / 2)
                b2.y -= ny * (overlap / 2)
                
                dvx = b1.vx - b2.vx
                dvy = b1.vy - b2.vy
                vel_along_normal = dvx * nx + dvy * ny
                
                if vel_along_normal > 0:
                    continue
                    
                j_impulse = -(1 + RESTITUTION) * vel_along_normal
                j_impulse /= 2 
                
                impulse_x = j_impulse * nx
                impulse_y = j_impulse * ny
                
                b1.vx += impulse_x
                b1.vy += impulse_y
                b2.vx -= impulse_x
                b2.vy -= impulse_y

# =============================================================================
# CORE SIMULATION RUNNER (HEADLESS & MULTIPROCESSING)
# =============================================================================
def run_single_trial(args):
    """Runs exactly one trial. Isolated so it can be passed to a CPU core."""
    trial_id, params, trial_steps = args
    vent_wall, vent_sub_wall, vent_internal, thermal_kick = params

    # CRITICAL: Re-seed the random generator for this specific core!
    random.seed()
    np.random.seed()

    # Initialize Environment
    blocks = []
    for _ in range(NUM_BLOCKS):
        blocks.append(Block(blocks))

    vents = [Vent(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]

    # Run Physics Loop
    for ticks in range(1, trial_steps + 1):
        if ticks % 10 == 0:
            for v in vents: v.update_wind(vent_wall, vent_sub_wall, vent_internal)

        for b in blocks:
            b.apply_vent_physics(vents, thermal_kick)
            b.update()
            
        check_collisions(blocks)

    # Calculate average distance traveled relative to the (15, 15) corner
    final_displacements = [get_distance_pixels(b.x, b.y) - get_distance_pixels(b.start_x, b.start_y) for b in blocks]
    return sum(final_displacements) / len(final_displacements)

def run_headless_trials(params, target_steps):
    """Fires off trials across multiple CPU cores simultaneously."""
    args_list = [(i, params, target_steps) for i in range(TRIALS_PER_EVALUATION)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=TRIALS_PER_EVALUATION) as executor:
        results = list(executor.map(run_single_trial, args_list))
        
    return sum(results) / len(results)

# =============================================================================
# OPTIMIZATION COST FUNCTION
# =============================================================================
evaluation_count = 0

def cost_function(params, target_disp, target_steps):
    global evaluation_count
    
    # Run the simulation (Triggers multiprocessing)
    mean_disp = run_headless_trials(params, target_steps)
    
    # Calculate the error (cost)
    cost = (mean_disp - target_disp)**2
    
    evaluation_count += 1
    print(f"Eval {evaluation_count:03d} | Params: [W:{params[0]:.4f}, SW:{params[1]:.4f}, I:{params[2]:.4f}, K:{params[3]:.4f}] | Yields: {mean_disp:+.2f}px | Diff: {(mean_disp - target_disp):+.2f}px")
    
    return cost

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    pygame.init()
    
    target_val = float(input("Enter TARGET DISPLACEMENT (px) from (15, 15) corner (e.g., 9.46): "))
    target_steps = int(input("Enter NUMBER OF STEPS to run before measuring (e.g., 701): "))
    
    print(f"\nStarting optimization for {target_val}px at {target_steps} steps.")
    print("This will utilize 5 CPU cores on your Mac. Please wait...\n")

    # Initial guess derived from your original True AO script
    # [VENT_STRENGTH_WALL, VENT_STRENGTH_SUB_WALL, VENT_STRENGTH_INTERNAL, THERMAL_KICK]
    initial_guess = [0.08, 0.0, 0.1, 1.6]

    # Run Nelder-Mead optimization
    result = minimize(
        cost_function, 
        initial_guess, 
        args=(target_val, target_steps), 
        method='Nelder-Mead',
        options={'maxiter': 150, 'xatol': 0.1, 'fatol': 0.1}
    )

    print("\n==================================================")
    if result.success or result.status == 2:  # Status 2 is maxiter reached
        opt = result.x
        print("OPTIMIZATION COMPLETE!")
        print(f"Target: {target_val} px at {target_steps} steps")
        print("--------------------------------------------------")
        print(f"VENT_STRENGTH_WALL     = {opt[0]:.5f}")
        print(f"VENT_STRENGTH_SUB_WALL = {opt[1]:.5f}")
        print(f"VENT_STRENGTH_INTERNAL = {opt[2]:.5f}")
        print(f"THERMAL_KICK           = {opt[3]:.5f}")
    else:
        print("Optimizer failed to converge.")
        print(result.message)
    print("==================================================")