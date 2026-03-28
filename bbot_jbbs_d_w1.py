# =============================================================================
# TRUE AO DEPLETION BROWNIAN HACK (SAMPLED)
# =============================================================================
import os
import sys
os.environ['PYTHON_APPLE_ALLOW_SDL2_MIX'] = '1'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
import math
import random
import csv
import statistics
from datetime import datetime
import matplotlib.pyplot as plt
import cv2          
import numpy as np  

# =============================================================================
# SECTION 1: GLOBAL CONFIGURATION (CONSTANTS)
# =============================================================================
WIDTH, HEIGHT = 600, 600          
BG_COLOR = (20, 20, 20)           
DARK_MARKER_COLOR = (101, 67, 33) 
LIGHT_POS = (15, 15)              
LIGHT_RADIUS_VISUAL = 5           
BLOCK_COLOR = (80, 80, 220)       
WALL_COLOR = (255, 255, 0)        
TEXT_COLOR = (255, 255, 255)      

NUM_BLOCKS = 20                   
BLOCK_RADIUS = 20                 
WALL_THICKNESS = 10               

# --- TIMING & SAMPLING ---
TRIAL_STEPS = 10000
SAMPLE_INTERVAL = 1000

# --- VENT & NOISE SETTINGS ---
GRID_SIZE = 10 
NUM_VENTS = GRID_SIZE * GRID_SIZE 

# TRUE AO CONTAINMENT FIELD
VENT_REACH = 60                   
BLOCK_FRICTION = 0.90             
RESTITUTION = 0.5

# trial 1
#VENT_STRENGTH_WALL     = 0.07871
#VENT_STRENGTH_SUB_WALL = 0.00016
#VENT_STRENGTH_INTERNAL = 0.10323
#THERMAL_KICK           = 1.53471

# trial 2
#OPTIMIZATION COMPLETE!
#Target: 8.232710950622184 px at 1000 steps
#--------------------------------------------------
VENT_STRENGTH_WALL     = 0.08000
VENT_STRENGTH_SUB_WALL = 0.00000
VENT_STRENGTH_INTERNAL = 0.09999
THERMAL_KICK           = 1.60027

# =============================================================================
# SECTION 2: HELPER FUNCTIONS & CLASSES
# =============================================================================
def save_screenshot(screen, folder, filename):
    path = os.path.join(folder, filename) 
    pygame.image.save(screen, path)

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
    return math.hypot(x - LIGHT_POS[0], y - LIGHT_POS[1])

def draw_text(screen, text, size, x, y, color=TEXT_COLOR):
    font = pygame.font.SysFont("Arial", size, bold=False)
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y)) 

# --- CLASSES ---
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
        
        self.update_wind()

    def update_wind(self):
        if self.is_boundary or self.is_sub_boundary:
            # TRUE AO LOGIC: Blow orthogonally into the nearest wall
            nx, ny = 0, 0
            
            if self.gx <= 1: nx = -1                
            elif self.gx >= GRID_SIZE - 2: nx = 1   
            
            if self.gy <= 1: ny = -1                
            elif self.gy >= GRID_SIZE - 2: ny = 1   
            
            dist = math.hypot(nx, ny)
            strength = VENT_STRENGTH_WALL if self.is_boundary else VENT_STRENGTH_SUB_WALL

            if dist > 0:
                # --- CORNER KICKER MODIFICATION ---
                if dist > 1.0: 
                    # It's a corner! 
                    # 1. Reverse the direction to blow OUT of the corner
                    nx = -nx 
                    ny = -ny
                    # 2. We deliberately DO NOT divide by dist here. 
                    # Leaving it unnormalized gives it a 1.41x stronger kick.
                else:
                    # It's a flat wall. Normalize it normally.
                    nx /= dist
                    ny /= dist
                    
            self.vx = nx * strength
            self.vy = ny * strength
            
        else:
            # Deep internal vents blow randomly (Turbulence)
            angle = random.uniform(0, 2 * math.pi)
            self.vx = math.cos(angle) * VENT_STRENGTH_INTERNAL
            self.vy = math.sin(angle) * VENT_STRENGTH_INTERNAL

    def draw(self, screen):
        # Color code the vents: Red = Strong Wall, Orange = Weak Sub-Wall, Green = Random
        if self.is_boundary:
            color = (255, 100, 100)
        elif self.is_sub_boundary:
            color = (255, 165, 0)
        else:
            color = (100, 255, 100)
            
        end_pos = (self.x + self.vx * 30, self.y + self.vy * 30)
        pygame.draw.line(screen, color, (self.x, self.y), end_pos, 1)
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 2)

class Block:
    def __init__(self, existing_blocks):
        self.radius = BLOCK_RADIUS
        self.x, self.y = get_valid_block_pos(existing_blocks)
        self.start_x, self.start_y = self.x, self.y 
        self.vx, self.vy = 0.0, 0.0 

    def apply_vent_physics(self, vents):
        self.vx += random.uniform(-THERMAL_KICK, THERMAL_KICK)
        self.vy += random.uniform(-THERMAL_KICK, THERMAL_KICK)

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
# SECTION 3: EXPORTING & PLOTTING
# =============================================================================
def export_sampled_results(all_trials_data, output_folder):
    csv_path = os.path.join(output_folder, "vent_sampled_results.csv")
    
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        sample_steps = sorted(list(all_trials_data[0]['time_series'].keys()))
        headers = ["Trial"] + [f"Step_{step}" for step in sample_steps]
        writer.writerow(headers)
        
        for r in all_trials_data:
            # Use .get(step, 0) instead of direct bracket access just to be safe
            row = [r['trial']] + [r['time_series'].get(step, 0) for step in sample_steps]
            writer.writerow(row)
            
        writer.writerow([])
        
        avg_row = ["OVERALL MEAN"]
        for step in sample_steps:
            # Safely get the step data
            step_mean = statistics.mean([r['time_series'].get(step, 0) for r in all_trials_data])
            avg_row.append(step_mean)    
        writer.writerow(avg_row)

def plot_histogram(data, title, xlabel, ylabel, filename, color, output_folder):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=40, color=color, edgecolor='black', alpha=0.7)
    mean_val = statistics.mean(data)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}px')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    sd_val = statistics.stdev(data) if len(data) > 1 else 0
    info_text = f"Pooled SD: {sd_val:.2f}px\nTotal Samples: {len(data)}"
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()

# =============================================================================
# SECTION 4: USER INTERFACE ROUTINES
# =============================================================================
def ui_ask_num_trials(screen):
    input_text = "100"
    while True:
        screen.fill(BG_COLOR)
        draw_text(screen, "SET NUMBER OF TRIALS", 28, WIDTH//2 - 150, 100, (100, 255, 255))
        draw_text(screen, "How many trials would you like to run?", 20, WIDTH//2 - 160, 160)
        draw_text(screen, f"(Running for {TRIAL_STEPS} steps each)", 16, WIDTH//2 - 110, 190, (150, 150, 150))
        
        box_rect = pygame.Rect(WIDTH//2 - 60, 230, 120, 50)
        pygame.draw.rect(screen, (255, 255, 255), box_rect)
        draw_text(screen, input_text, 32, box_rect.x + 15, box_rect.y + 10, (0, 0, 0))
        
        draw_text(screen, "Type number and press [ENTER] to begin.", 18, WIDTH//2 - 160, 310, (200, 200, 200))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE: input_text = input_text[:-1]
                elif event.unicode.isnumeric(): input_text += event.unicode
                elif event.key == pygame.K_RETURN and input_text:
                    return int(input_text)

def ui_show_headless_progress(screen, current, total):
    screen.fill(BG_COLOR)
    draw_text(screen, "RUNNING HEADLESS SIMULATIONS", 28, WIDTH//2 - 200, HEIGHT//2 - 40, (255, 215, 0))
    draw_text(screen, f"Crunching Trial {current} / {total}...", 20, WIDTH//2 - 100, HEIGHT//2 + 10)
    pygame.display.flip()
    pygame.event.pump() 

# =============================================================================
# SECTION 5: SIMULATION RUNNER (FIXED)
# =============================================================================
def run_simulation(trial_num, is_visual, output_folder, screen=None):
    blocks = []

    for _ in range(NUM_BLOCKS):
        blocks.append(Block(blocks))

    vents = [Vent(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        
    ticks = 0
    video_writer = None 
    time_series_data = {}
    
    # We create the clock NO MATTER WHAT, even for headless
    clock = pygame.time.Clock()
        
    if is_visual and screen:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_writer = cv2.VideoWriter(os.path.join(output_folder, "00_trial_01_video.mp4"), fourcc, 60.0, (WIDTH, HEIGHT))
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN: waiting = False
            screen.fill(BG_COLOR)
            draw_text(screen, f"True AO Vent + Brownian Hack", 32, WIDTH//2 - 200, 100, (100, 255, 100))
            draw_text(screen, "Press ENTER to Start Trial 1 (Visual)", 24, WIDTH//2 - 180, 160, (255, 255, 0))
            pygame.display.flip()
            
        save_screenshot(screen, output_folder, "01_sim_start.png")
        last_shot_time = 0

    running = True
    while running:
        # ALWAYS pump events so Pygame doesn't deadlock, even in headless
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        ticks += 1
        sim_time = ticks / 60.0 
        
        # 1. Update internal vents (Turbulence)
        if ticks % 10 == 0:
            for v in vents: v.update_wind()

        # 2. Apply Physics & Collisions
        for b in blocks:
            b.apply_vent_physics(vents)
            b.update()
        check_collisions(blocks) 
        
        # 3. Time-Series Sampling
        if ticks % SAMPLE_INTERVAL == 0:
            rock_displacements = [get_distance_pixels(b.x, b.y) - get_distance_pixels(b.start_x, b.start_y) for b in blocks]
            time_series_data[ticks] = statistics.mean(rock_displacements)

        # 4. End Condition
        if ticks >= TRIAL_STEPS:
            running = False 
            
        if is_visual and screen:
            screen.fill(BG_COLOR)
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, HEIGHT-WALL_THICKNESS, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WALL_THICKNESS, HEIGHT))
            pygame.draw.rect(screen, WALL_COLOR, (WIDTH-WALL_THICKNESS, 0, WALL_THICKNESS, HEIGHT))
            pygame.draw.circle(screen, DARK_MARKER_COLOR, LIGHT_POS, LIGHT_RADIUS_VISUAL)
            
            draw_text(screen, f"{sim_time:.2f}s | {ticks} / {TRIAL_STEPS} steps", 20, 10, 80, (255, 255, 0))
            
            if time_series_data:
                latest_sample = max(time_series_data.keys())
                disp_text = f"Disp (@{latest_sample}): {time_series_data[latest_sample]:+.1f} px"
                draw_text(screen, disp_text, 20, WIDTH - 220, 80, (150, 255, 150))

            for v in vents: v.draw(screen)
            for b in blocks: pygame.draw.circle(screen, BLOCK_COLOR, (int(b.x), int(b.y)), int(b.radius))
                
            pygame.display.flip() 
            if video_writer is not None:
                frame = pygame.surfarray.array3d(screen).transpose([1, 0, 2])
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            if sim_time - last_shot_time >= 5:
                save_screenshot(screen, output_folder, f"02_progress_{int(sim_time)}s.png")
                last_shot_time = sim_time
                
            # Cap visual framerate at 60fps
            clock.tick(60) 
            
        else:
            # HEADLESS MODE: Run as fast as possible, but yield slightly so OS doesn't hang
            clock.tick(10000)

    if video_writer is not None: video_writer.release() 
    if is_visual and screen: save_screenshot(screen, output_folder, "03_sim_finished.png")

    final_displacements = [get_distance_pixels(b.x, b.y) - get_distance_pixels(b.start_x, b.start_y) for b in blocks]
    
    return {
        'trial': trial_num, 
        'time_series': time_series_data,
        'final_displacements': final_displacements
    }

# =============================================================================
# SECTION 6: MASTER CONTROL
# =============================================================================
def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("TRUE AO VENT BATCH")

    num_trials = ui_ask_num_trials(screen)

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = f"vent_hack_true_ao_{timestamp_str}"
    os.makedirs(output_folder, exist_ok=True)
    
    all_results = [run_simulation(1, True, output_folder, screen=screen)]
    
    if num_trials > 1:
        for i in range(2, num_trials + 1):
            if i % max(1, (num_trials // 20)) == 0 or i == 2: 
                ui_show_headless_progress(screen, i, num_trials)
            all_results.append(run_simulation(i, False, output_folder))

    pooled_final_displacements = []
    for r in all_results:
        pooled_final_displacements.extend(r['final_displacements'])
            
    export_sampled_results(all_results, output_folder)
    
    plot_histogram(pooled_final_displacements, 'Final Rock Displacements (Step 10000)', 'Net Displacement (pixels)', 'Frequency', "04_final_displacement_hist.png", 'forestgreen', output_folder)
    
    # Final screen
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN):
                waiting = False
        screen.fill(BG_COLOR)
        draw_text(screen, "BATCH COMPLETE", 32, WIDTH//2 - 120, HEIGHT//2 - 60, (60, 220, 60))
        draw_text(screen, f"Sampled Data and Histograms saved to:", 18, WIDTH//2 - 150, HEIGHT//2)
        draw_text(screen, f"{output_folder}", 16, WIDTH//2 - 180, HEIGHT//2 + 30, (200, 200, 200))
        draw_text(screen, "Press [ENTER] to exit.", 18, WIDTH//2 - 80, HEIGHT//2 + 80, (255, 255, 0))
        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__":
    main()