import pygame
import math
import random
import sys
import os
import csv
import statistics
from datetime import datetime
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 1: GLOBAL CONFIGURATION
# =============================================================================

WIDTH, HEIGHT = 600, 600          
BG_COLOR = (20, 20, 20)           
LIGHT_COLOR = (255, 255, 240)     
LIGHT_POS = (30, 30)              
LIGHT_RADIUS_VISUAL = 40          

ROBOT_COLOR = (220, 60, 60)       
REST_COLOR = (60, 220, 60)        
BLOCK_COLOR = (80, 80, 220)       
WALL_COLOR = (255, 255, 0)        
TEXT_COLOR = (255, 255, 255)      

NUM_ROBOTS = 20                   
NUM_BLOCKS = 30                   
BLOCK_RADIUS = 13                 
BLOCK_SIZE = 26                   
ROBOT_RADIUS = 10                 
WALL_THICKNESS = 10               

SHADOW_PROXIMITY_LIMIT = 20       
ROBOT_SPEED_GO = 3.5              
ROBOT_SPEED_PUSH = 1.0            
PUSH_FORCE = 1.0                  
BLOCK_FRICTION = 0.90             

SIM_TIMEOUT = 300.0               

# =============================================================================
# SECTION 2: HELPER FUNCTIONS & CLASSES
# =============================================================================

def save_screenshot(screen, folder, filename):
    path = os.path.join(folder, filename)
    pygame.image.save(screen, path)

def get_robot_spawn_pos():
    safe_min = WALL_THICKNESS + ROBOT_RADIUS
    safe_max_x = WIDTH - WALL_THICKNESS - ROBOT_RADIUS
    safe_max_y = HEIGHT - WALL_THICKNESS - ROBOT_RADIUS
    while True: 
        x = random.uniform(safe_min, safe_max_x)
        y = random.uniform(safe_min, safe_max_y)
        if math.hypot(x - LIGHT_POS[0], y - LIGHT_POS[1]) > LIGHT_RADIUS_VISUAL:
            return x, y

def get_valid_block_pos(existing_robots):
    safe_min = WALL_THICKNESS + BLOCK_RADIUS
    safe_max_x = WIDTH - WALL_THICKNESS - BLOCK_RADIUS
    safe_max_y = HEIGHT - WALL_THICKNESS - BLOCK_RADIUS
    while True: 
        x = random.randint(int(safe_min), int(safe_max_x))
        y = random.randint(int(safe_min), int(safe_max_y))
        if math.hypot(x - LIGHT_POS[0], y - LIGHT_POS[1]) < (LIGHT_RADIUS_VISUAL + BLOCK_RADIUS + 5):
            continue 
        if any(math.hypot(x - r.x, y - r.y) < (BLOCK_RADIUS + ROBOT_RADIUS + 2) for r in existing_robots):
            continue
        return x, y

def get_distance_pixels(x, y):
    return math.hypot(x - LIGHT_POS[0], y - LIGHT_POS[1])

def draw_text(screen, text, size, x, y, color=TEXT_COLOR):
    font = pygame.font.SysFont("Arial", size, bold=False)
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))

class Robot:
    def __init__(self):
        self.x, self.y = get_robot_spawn_pos() 
        self.angle = random.uniform(0, 2 * math.pi)
        self.touching_block = False 
        self.in_shadow = False
        self.stop_time = None 

    def sense(self, blocks, light_pos):
        self.in_shadow = False
        self.touching_block = False
        dx, dy = self.x - light_pos[0], self.y - light_pos[1]
        dist_to_light = math.hypot(dx, dy)
        if dist_to_light == 0: return 

        dir_x, dir_y = -(dx / dist_to_light), -(dy / dist_to_light)
        sensor_x = self.x + (dir_x * ROBOT_RADIUS)
        sensor_y = self.y + (dir_y * ROBOT_RADIUS)
        line_start, line_end = light_pos, (sensor_x, sensor_y)
        
        for block in blocks:
            eff_rad = block.radius if block.shape == "CIRCLE" else (block.w / 2)
            bx, by = block.x - eff_rad, block.y - eff_rad
            rect = pygame.Rect(int(bx), int(by), int(eff_rad * 2), int(eff_rad * 2))
            
            if rect.clipline(line_start, line_end):
                dist_robot_block = math.hypot(self.x - block.x, self.y - block.y)
                if dist_robot_block <= (SHADOW_PROXIMITY_LIMIT + ROBOT_RADIUS + eff_rad):
                    self.in_shadow = True
                    break

    def decide_and_move(self):
        if self.in_shadow: return 
        speed = ROBOT_SPEED_PUSH if self.touching_block else ROBOT_SPEED_GO
        if not self.touching_block: self.angle += random.uniform(-0.4, 0.4) 

        self.x += math.cos(self.angle) * speed
        self.y += math.sin(self.angle) * speed
        min_bound = WALL_THICKNESS + ROBOT_RADIUS
        self.x = max(min_bound, min(WIDTH - min_bound, self.x))
        self.y = max(min_bound, min(HEIGHT - min_bound, self.y))

class Block:
    def __init__(self, existing_robots, shape_type):
        self.shape = shape_type 
        self.radius = BLOCK_RADIUS
        self.w, self.h = BLOCK_SIZE, BLOCK_SIZE
        self.x, self.y = get_valid_block_pos(existing_robots)
        self.start_x, self.start_y = self.x, self.y
        self.vx, self.vy = 0.0, 0.0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= BLOCK_FRICTION
        self.vy *= BLOCK_FRICTION
        
        half_size = self.radius if self.shape == "CIRCLE" else (self.w / 2)
        min_bound = WALL_THICKNESS + half_size
        if self.x < min_bound: self.x = min_bound; self.vx = 0
        if self.x > WIDTH - min_bound: self.x = WIDTH - min_bound; self.vx = 0
        if self.y < min_bound: self.y = min_bound; self.vy = 0
        if self.y > HEIGHT - min_bound: self.y = HEIGHT - min_bound; self.vy = 0

def check_collisions(robots, blocks):
    for robot in robots:
        for block in blocks:
            eff_radius = block.radius if block.shape == "CIRCLE" else (block.w / 2)
            dx, dy = block.x - robot.x, block.y - robot.y
            dist = math.hypot(dx, dy)
            if dist < (ROBOT_RADIUS + eff_radius):
                robot.touching_block = True
                if dist < 0.1: dist = 0.1 
                nx, ny = dx / dist, dy / dist
                block.vx += nx * PUSH_FORCE
                block.vy += ny * PUSH_FORCE
                robot.x -= nx * 2; robot.y -= ny * 2

    for i, b1 in enumerate(blocks):
        for b2 in blocks[i+1:]:
            dx, dy = b1.x - b2.x, b1.y - b2.y
            dist = math.hypot(dx, dy)
            r1 = b1.radius if b1.shape == "CIRCLE" else (b1.w / 2)
            r2 = b2.radius if b2.shape == "CIRCLE" else (b2.w / 2)
            if dist < (r1 + r2):
                if dist < 0.1: dist = 0.1
                nx, ny = dx / dist, dy / dist
                b1.x += nx * 2.0; b1.y += ny * 2.0
                b2.x -= nx * 2.0; b2.y -= ny * 2.0

# =============================================================================
# SECTION 3: EXPORTING, REPORTING & PLOTTING
# =============================================================================

def export_trial1_metrics(blocks, robots, duration, avg_change, sd_change, output_folder):
    filename = os.path.join(output_folder, "trial_01_metrics.csv")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["--- TRIAL 1 ROCK DATA ---"])
        writer.writerow(["Rock ID", "Start Dist (px)", "End Dist (px)", "Net Change (px)"])
        for i, b in enumerate(blocks):
            d_start = get_distance_pixels(b.start_x, b.start_y)
            d_end = get_distance_pixels(b.x, b.y)
            writer.writerow([i+1, round(d_start, 2), round(d_end, 2), round(d_end - d_start, 2)])
        writer.writerow([])
        writer.writerow(["--- TRIAL 1 BOT DATA ---"])
        writer.writerow(["Bot ID", "Stop Time (sec)"])
        for i, r in enumerate(robots):
            writer.writerow([i+1, round(r.stop_time if r.stop_time else duration, 2)])
            
def export_aggregate_summary(all_results, output_folder, mean_bot_sds, sd_pooled_bots, mean_rock_sds, sd_pooled_rocks):
    filename = os.path.join(output_folder, "aggregate_summary.csv")
    
    # NEW: Find trials where mean distance moved is <= 0
    zero_or_less_trials = [str(r['trial']) for r in all_results if r['avg_rock_change'] <= 0]
    trials_str = ", ".join(zero_or_less_trials) if zero_or_less_trials else "None"

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the new metric at the very top
        writer.writerow(["Trials with Mean Rock Change <= 0", trials_str])
        writer.writerow([])
        
        writer.writerow(["Total Trials", len(all_results)])
        avg_dur = statistics.mean([r['duration'] for r in all_results])
        avg_net_rock = statistics.mean([r['avg_rock_change'] for r in all_results])
        
        writer.writerow(["Avg Sim Duration (s)", round(avg_dur, 2)])
        writer.writerow(["Avg Net Rock Change (px)", round(avg_net_rock, 2)])
        writer.writerow(["Mean of Intra-Run Bot Stop SDs (s)", round(mean_bot_sds, 2)])
        writer.writerow(["SD of Pooled Bot Stop Times (s)", round(sd_pooled_bots, 2)])
        writer.writerow(["Mean of Intra-Run Rock Dist SDs (px)", round(mean_rock_sds, 2)])
        writer.writerow(["SD of Pooled Rock Distances (px)", round(sd_pooled_rocks, 2)])
        writer.writerow([])
        
        writer.writerow(["Trial", "Duration", "Avg Net Rock Change", "Bot Stop SD", "Rock Change SD"])
        for r in all_results:
            st = r['stop_times']
            rc = r['rock_changes']
            writer.writerow([
                r['trial'], 
                round(r['duration'], 2), 
                round(r['avg_rock_change'], 2), 
                round(statistics.stdev(st) if len(st) > 1 else 0, 2),
                round(statistics.stdev(rc) if len(rc) > 1 else 0, 2)
            ])

def plot_bot_histogram(pooled_times, mean_of_sds, sd_of_pooled, output_folder):
    plt.figure(figsize=(10, 6))
    plt.hist(pooled_times, bins=40, color='royalblue', edgecolor='black', alpha=0.7)
    mean_val = statistics.mean(pooled_times)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean Time: {mean_val:.2f}s')
    plt.title('Distribution of Bot Stop Times Across All Runs', fontsize=14, fontweight='bold')
    plt.xlabel('Simulation Time (seconds)', fontsize=12)
    plt.ylabel('Frequency (Number of Bots)', fontsize=12)
    info_text = f"Total Pooled SD: {sd_of_pooled:.2f}s\nMean of Intra-run SDs: {mean_of_sds:.2f}s\nTotal Bots: {len(pooled_times)}"
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "05_variance_bot_histogram.png"))
    plt.close()

def plot_rock_histogram(pooled_rock_changes, mean_rock_sds, sd_pooled_rocks, output_folder):
    plt.figure(figsize=(10, 6))
    plt.hist(pooled_rock_changes, bins=40, color='forestgreen', edgecolor='black', alpha=0.7)
    mean_val = statistics.mean(pooled_rock_changes)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean Change: {mean_val:.2f}px')
    plt.title('Distribution of Net Rock Displacements Across All Runs', fontsize=14, fontweight='bold')
    plt.xlabel('Net Displacement (pixels)', fontsize=12)
    plt.ylabel('Frequency (Number of Rocks)', fontsize=12)
    info_text = f"Total Pooled SD: {sd_pooled_rocks:.2f}px\nMean of Intra-run SDs: {mean_rock_sds:.2f}px\nTotal Rocks: {len(pooled_rock_changes)}"
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "06_variance_rock_histogram.png"))
    plt.close()

# =============================================================================
# SECTION 4: USER INTERFACE ROUTINES
# =============================================================================

def ui_get_num_trials(screen):
    clock = pygame.time.Clock()
    input_text = ""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return max(1, int(input_text)) if input_text.strip() else 1
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                elif event.unicode.isnumeric():
                    input_text += event.unicode

        screen.fill(BG_COLOR)
        draw_text(screen, "--- BRAITENBERG SWARM AUTOMATOR ---", 24, WIDTH//2 - 180, HEIGHT//2 - 60, (255, 215, 0))
        draw_text(screen, "Enter number of trials to simulate:", 20, WIDTH//2 - 140, HEIGHT//2 - 20)
        draw_text(screen, input_text + "_", 32, WIDTH//2 - 20, HEIGHT//2 + 20, (100, 255, 100))
        pygame.display.flip()
        clock.tick(30)

def ui_show_headless_progress(screen, current, total):
    screen.fill(BG_COLOR)
    draw_text(screen, "RUNNING HEADLESS SIMULATIONS", 28, WIDTH//2 - 200, HEIGHT//2 - 40, (255, 215, 0))
    draw_text(screen, f"Crunching Trial {current} / {total}...", 20, WIDTH//2 - 100, HEIGHT//2 + 10)
    draw_text(screen, "Please wait. This may take a moment.", 16, WIDTH//2 - 130, HEIGHT//2 + 50, (150, 150, 150))
    pygame.display.flip()
    pygame.event.pump() 

def ui_show_final_done(screen, output_folder):
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN):
                waiting = False
        screen.fill(BG_COLOR)
        draw_text(screen, "BATCH COMPLETE", 32, WIDTH//2 - 120, HEIGHT//2 - 60, (60, 220, 60))
        draw_text(screen, f"Data and 2 Histograms successfully saved to:", 18, WIDTH//2 - 160, HEIGHT//2)
        draw_text(screen, f"{output_folder}", 16, WIDTH//2 - 180, HEIGHT//2 + 30, (200, 200, 200))
        draw_text(screen, "Press [ENTER] to exit.", 18, WIDTH//2 - 80, HEIGHT//2 + 80, (255, 255, 0))
        pygame.display.flip()
        pygame.time.Clock().tick(15)

def ui_show_trial1_results(screen, duration, avg_change, sd_change, output_folder, mean_stop, sd_stop):
    waiting, screenshot_taken = True, False
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN: waiting = False

        screen.fill(BG_COLOR)
        draw_text(screen, "TRIAL 1 COMPLETE", 32, 30, 20, (255, 215, 0))
        draw_text(screen, f"Duration:       {duration:.2f} s", 18, 30, 70)
        draw_text(screen, f"Mean Stop Time: {mean_stop:.2f} s", 18, 30, 95)
        draw_text(screen, f"SD of Stops:    {sd_stop:.2f} s", 18, 30, 120, (200, 200, 200))
        sign = "" if avg_change < 0 else "+"
        draw_text(screen, f"Rock Change:    {sign}{avg_change:.2f} px", 18, 320, 70)
        draw_text(screen, f"Rock Change SD: {sd_change:.2f} px", 18, 320, 95, (200, 200, 200))
        draw_text(screen, "Press [ENTER] to begin Headless Runs.", 22, 30, HEIGHT - 50, (255, 100, 100))

        pygame.display.flip()
        if not screenshot_taken:
            save_screenshot(screen, output_folder, "04_final_metrics.png")
            screenshot_taken = True
        pygame.time.Clock().tick(15)

# =============================================================================
# SECTION 5: SIMULATION RUNNER
# =============================================================================

def run_simulation(trial_num, is_visual, output_folder, screen=None):
    robots = [Robot() for _ in range(NUM_ROBOTS)]
    blocks = [Block(robots, "CIRCLE" if i < (NUM_BLOCKS//2) else "SQUARE") for i in range(NUM_BLOCKS)]
    ticks, sim_time, final_time = 0, 0.0, None
    
    if is_visual and screen:
        pygame.display.set_caption("Trial 1: Visual Mode")
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
            for b in blocks:
                if b.shape == "CIRCLE": pygame.draw.circle(screen, BLOCK_COLOR, (int(b.x), int(b.y)), int(b.radius))
                else: pygame.draw.rect(screen, BLOCK_COLOR, (int(b.x - b.w/2), int(b.y - b.h/2), b.w, b.h))
            for r in robots: pygame.draw.circle(screen, ROBOT_COLOR, (int(r.x), int(r.y)), ROBOT_RADIUS)
            draw_text(screen, "Press ENTER to Start Trial 1", 32, WIDTH//2 - 170, 100, (255, 255, 0))
            pygame.display.flip()
            clock.tick(60)
        save_screenshot(screen, output_folder, "01_sim_start.png")
        last_shot_time = 0

    running = True
    while running:
        if is_visual:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()

        ticks += 1
        sim_time = ticks / 60.0  
        
        check_collisions(robots, blocks)
        for b in blocks: b.update() 
        
        safe_robots = 0
        for r in robots:
            r.sense(blocks, LIGHT_POS)
            r.decide_and_move()
            if r.in_shadow:
                safe_robots += 1
                if r.stop_time is None: r.stop_time = sim_time
            else:
                r.stop_time = None
                
        if safe_robots == NUM_ROBOTS or sim_time >= SIM_TIMEOUT:
            final_time = sim_time
            running = False
            
        if is_visual and screen:
            screen.fill(BG_COLOR)
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, HEIGHT-WALL_THICKNESS, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WALL_THICKNESS, HEIGHT))
            pygame.draw.rect(screen, WALL_COLOR, (WIDTH-WALL_THICKNESS, 0, WALL_THICKNESS, HEIGHT))
            pygame.draw.circle(screen, LIGHT_COLOR, LIGHT_POS, LIGHT_RADIUS_VISUAL)
            draw_text(screen, f"{sim_time:.2f}s", 20, 10, 80, (255, 255, 0))
            for b in blocks:
                if b.shape == "CIRCLE": pygame.draw.circle(screen, BLOCK_COLOR, (int(b.x), int(b.y)), int(b.radius))
                else: pygame.draw.rect(screen, BLOCK_COLOR, (int(b.x - b.w/2), int(b.y - b.h/2), b.w, b.h))
            for r in robots:
                color = REST_COLOR if r.in_shadow else ROBOT_COLOR
                pygame.draw.circle(screen, color, (int(r.x), int(r.y)), ROBOT_RADIUS)
            pygame.display.flip()
            if sim_time - last_shot_time >= 5:
                save_screenshot(screen, output_folder, f"02_progress_{int(sim_time)}s.png")
                last_shot_time = sim_time
            clock.tick(60)

    rock_changes = [get_distance_pixels(b.x, b.y) - get_distance_pixels(b.start_x, b.start_y) for b in blocks]
    avg_change = statistics.mean(rock_changes)
    sd_change = statistics.stdev(rock_changes) if len(rock_changes) > 1 else 0.0
    stop_times = [r.stop_time if r.stop_time is not None else final_time for r in robots]

    if is_visual and screen:
        save_screenshot(screen, output_folder, "03_sim_finished.png")
        export_trial1_metrics(blocks, robots, final_time, avg_change, sd_change, output_folder)
        m_stop = statistics.mean(stop_times) if stop_times else 0.0
        sd_stop = statistics.stdev(stop_times) if len(stop_times) > 1 else 0.0
        ui_show_trial1_results(screen, final_time, avg_change, sd_change, output_folder, m_stop, sd_stop)

    return {
        'trial': trial_num, 
        'duration': final_time, 
        'avg_rock_change': avg_change, 
        'rock_changes': rock_changes,
        'stop_times': stop_times
    }

# =============================================================================
# SECTION 6: MASTER CONTROL
# =============================================================================
def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Braitenberg Swarm Automator")

    num_trials = ui_get_num_trials(screen)

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = f"simulation_output_{timestamp_str}"
    os.makedirs(output_folder, exist_ok=True)
    
    all_results = [run_simulation(1, is_visual=True, output_folder=output_folder, screen=screen)]
    
    if num_trials > 1:
        for i in range(2, num_trials + 1):
            if i % 10 == 0 or i == 2: 
                ui_show_headless_progress(screen, i, num_trials)
            all_results.append(run_simulation(i, is_visual=False, output_folder=output_folder))

    intra_run_bot_sds, pooled_bot_times = [], []
    intra_run_rock_sds, pooled_rock_changes = [], []
    
    for r in all_results:
        st = r['stop_times']
        rc = r['rock_changes']
        pooled_bot_times.extend(st)
        pooled_rock_changes.extend(rc)
        if len(st) > 1: intra_run_bot_sds.append(statistics.stdev(st))
        if len(rc) > 1: intra_run_rock_sds.append(statistics.stdev(rc))
            
    mean_bot_sds = statistics.mean(intra_run_bot_sds) if intra_run_bot_sds else 0.0
    sd_pooled_bots = statistics.stdev(pooled_bot_times) if len(pooled_bot_times) > 1 else 0.0
    
    mean_rock_sds = statistics.mean(intra_run_rock_sds) if intra_run_rock_sds else 0.0
    sd_pooled_rocks = statistics.stdev(pooled_rock_changes) if len(pooled_rock_changes) > 1 else 0.0

    export_aggregate_summary(all_results, output_folder, mean_bot_sds, sd_pooled_bots, mean_rock_sds, sd_pooled_rocks)
    
    plot_bot_histogram(pooled_bot_times, mean_bot_sds, sd_pooled_bots, output_folder)
    plot_rock_histogram(pooled_rock_changes, mean_rock_sds, sd_pooled_rocks, output_folder)
    
    ui_show_final_done(screen, output_folder)
    pygame.quit()

if __name__ == "__main__":
    main()