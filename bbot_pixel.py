import pygame   # Import the library used for graphics and game loops
import math     # Import math library for calculating distances and angles
import random   # Import library to generate random numbers (for positions)
import sys      # Import system library to help exit the program cleanly
import time     # Import library to track how many seconds pass
import os       # Import operating system library to manage folders and files
import csv      # Import CSV library for exporting data
import statistics # Import statistics for Standard Deviation and Mean
from datetime import datetime # Import datetime to timestamp the folder

# =============================================================================
# SECTION 1: GLOBAL CONFIGURATION
# =============================================================================

# -- SCENARIO SETTINGS --
BLOCK_SPAWN_MODE = "RANDOM" 
ROBOT_SPAWN_MODE = "RANDOM"

# -- WINDOW SETTINGS --
WIDTH, HEIGHT = 600, 600          
BG_COLOR = (20, 20, 20)           
LIGHT_COLOR = (255, 255, 240)     

# -- LIGHT POSITION --
LIGHT_POS = (30, 30)              
LIGHT_RADIUS_VISUAL = 40          

# -- COLORS --
ROBOT_COLOR = (220, 60, 60)       
REST_COLOR = (60, 220, 60)        
BLOCK_COLOR = (80, 80, 220)       
WALL_COLOR = (255, 255, 0)        
TEXT_COLOR = (255, 255, 255)      

# -- ENTITIES --
NUM_ROBOTS = 20                   
NUM_BLOCKS = 30                   
BLOCK_RADIUS = 13                 
BLOCK_SIZE = 26                   
ROBOT_RADIUS = 10                 
WALL_THICKNESS = 10               

# -- LOGIC & PHYSICS --
MIN_SAFE_DISTANCE = 400           
SHADOW_PROXIMITY_LIMIT = 20       
ROBOT_SPEED_GO = 3.5              
ROBOT_SPEED_PUSH = 1.0            
PUSH_FORCE = 1.0                  
BLOCK_FRICTION = 0.90             

# -- OUTPUT SETTINGS --
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
OUTPUT_FOLDER = f"simulation_output_{timestamp_str}"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created output folder: {OUTPUT_FOLDER}")

# =============================================================================
# SECTION 2: HELPER FUNCTIONS
# =============================================================================

def save_screenshot(screen, filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    pygame.image.save(screen, path)
    print(f"Screenshot saved: {path}")

def get_robot_spawn_pos():
    safe_min = WALL_THICKNESS + ROBOT_RADIUS
    safe_max_x = WIDTH - WALL_THICKNESS - ROBOT_RADIUS
    safe_max_y = HEIGHT - WALL_THICKNESS - ROBOT_RADIUS

    while True: 
        x = random.uniform(safe_min, safe_max_x)
        y = random.uniform(safe_min, safe_max_y)
        
        dist_to_light = math.hypot(x - LIGHT_POS[0], y - LIGHT_POS[1])
        if dist_to_light > LIGHT_RADIUS_VISUAL:
            return x, y

def get_valid_block_pos(existing_robots):
    safe_min = WALL_THICKNESS + BLOCK_RADIUS
    safe_max_x = WIDTH - WALL_THICKNESS - BLOCK_RADIUS
    safe_max_y = HEIGHT - WALL_THICKNESS - BLOCK_RADIUS

    while True: 
        x = random.randint(int(safe_min), int(safe_max_x))
        y = random.randint(int(safe_min), int(safe_max_y))
        
        dist_to_light = math.hypot(x - LIGHT_POS[0], y - LIGHT_POS[1])
        if dist_to_light < (LIGHT_RADIUS_VISUAL + BLOCK_RADIUS + 5):
            continue 
            
        collision_found = False
        for r in existing_robots:
            dist = math.hypot(x - r.x, y - r.y)
            if dist < (BLOCK_RADIUS + ROBOT_RADIUS + 2): 
                collision_found = True
                break
        
        if collision_found:
            continue

        return x, y

# =============================================================================
# SECTION 3: CLASS DEFINITIONS
# =============================================================================

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
        
        dx = self.x - light_pos[0]
        dy = self.y - light_pos[1]
        dist_to_light = math.hypot(dx, dy)
        
        if dist_to_light == 0: return 

        if dist_to_light < MIN_SAFE_DISTANCE:
            self.in_shadow = False 
            return 

        dir_x = -(dx / dist_to_light)
        dir_y = -(dy / dist_to_light)
        sensor_x = self.x + (dir_x * ROBOT_RADIUS)
        sensor_y = self.y + (dir_y * ROBOT_RADIUS)
        
        line_start = light_pos
        line_end = (sensor_x, sensor_y)
        
        for block in blocks:
            if block.shape == "CIRCLE":
                bx = block.x - block.radius
                by = block.y - block.radius
                size = block.radius * 2
            else:
                bx = block.x - (block.w / 2)
                by = block.y - (block.h / 2)
                size = block.w
            
            rect = pygame.Rect(int(bx), int(by), int(size), int(size))
            
            if rect.clipline(line_start, line_end):
                dist_robot_block = math.hypot(self.x - block.x, self.y - block.y)
                required_dist = SHADOW_PROXIMITY_LIMIT + ROBOT_RADIUS + (block.radius if block.shape == "CIRCLE" else block.w/2)
                
                if dist_robot_block <= required_dist:
                    self.in_shadow = True
                    break

    def decide_and_move(self):
        if self.in_shadow:
            return 
        
        if self.touching_block:
            speed = ROBOT_SPEED_PUSH
        else:
            speed = ROBOT_SPEED_GO
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
    def __init__(self, existing_robots, shape_type):
        self.shape = shape_type 
        self.radius = BLOCK_RADIUS
        self.w = BLOCK_SIZE
        self.h = BLOCK_SIZE
        
        self.x, self.y = get_valid_block_pos(existing_robots)
        
        self.start_x = self.x
        self.start_y = self.y
        
        self.vx = 0.0
        self.vy = 0.0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        
        self.vx *= BLOCK_FRICTION
        self.vy *= BLOCK_FRICTION
        
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

def check_collisions(robots, blocks):
    for robot in robots:
        for block in blocks:
            eff_radius = block.radius if block.shape == "CIRCLE" else (block.w / 2)
            
            dx = block.x - robot.x
            dy = block.y - robot.y
            dist = math.hypot(dx, dy)
            
            if dist < (ROBOT_RADIUS + eff_radius):
                robot.touching_block = True
                if dist < 0.1: dist = 0.1 
                
                nx = dx / dist
                ny = dy / dist
                
                block.vx += nx * PUSH_FORCE
                block.vy += ny * PUSH_FORCE
                
                robot.x -= nx * 2
                robot.y -= ny * 2

    for i, b1 in enumerate(blocks):
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
                b1.x += nx * push; b1.y += ny * push
                b2.x -= nx * push; b2.y -= ny * push

def draw_text(screen, text, size, x, y, color=TEXT_COLOR):
    font = pygame.font.SysFont("Arial", size, bold=False)
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))

def get_distance_pixels(x, y):
    dx = x - LIGHT_POS[0]
    dy = y - LIGHT_POS[1]
    return math.hypot(dx, dy)

def export_metrics_to_csv(blocks, robots, duration, avg_start, avg_end, avg_change, sd_change, mean_stop, sd_stop):
    filename = os.path.join(OUTPUT_FOLDER, "metrics_report.csv")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write Global Summary
        writer.writerow(["--- GLOBAL SUMMARY ---"])
        writer.writerow(["Sim Duration (sec)", round(duration, 2)])
        writer.writerow(["Avg Start Distance (px)", round(avg_start, 2)])
        writer.writerow(["Avg End Distance (px)", round(avg_end, 2)])
        writer.writerow(["Net Avg Change (px)", round(avg_change, 2)])
        writer.writerow(["SD of Distance Change (px)", round(sd_change, 2)])
        writer.writerow(["Mean Stop Time (sec)", round(mean_stop, 2)])
        writer.writerow(["SD of Stop Times (sec)", round(sd_stop, 2)])
        writer.writerow([]) 
        
        # Write Granular Block Data
        writer.writerow(["--- BLOCK DATA ---"])
        writer.writerow(["Block ID", "Start Distance (px)", "End Distance (px)", "Net Change (px)"])
        for i, b in enumerate(blocks):
            d_start = get_distance_pixels(b.start_x, b.start_y)
            d_end = get_distance_pixels(b.x, b.y)
            change = d_end - d_start
            writer.writerow([i+1, round(d_start, 2), round(d_end, 2), round(change, 2)])
            
        writer.writerow([]) 
        
        # Write Granular Robot Data
        writer.writerow(["--- BROWNIAN BOT DATA ---"])
        writer.writerow(["Bot ID", "Final Stop Time (sec)"])
        for i, r in enumerate(robots):
            stop_t = r.stop_time if r.stop_time is not None else duration
            writer.writerow([i+1, round(stop_t, 2)])
            
    print(f"Data successfully exported to: {filename}")

def show_results_screen(screen, blocks, robots, duration):
    total_start_dist = 0
    total_end_dist = 0
    changes = []
    block_stats = []

    # Process Block Data
    for i, b in enumerate(blocks):
        d_start = get_distance_pixels(b.start_x, b.start_y)
        d_end = get_distance_pixels(b.x, b.y)
        change = d_end - d_start
        changes.append(change)
        total_start_dist += d_start
        total_end_dist += d_end
        block_stats.append(f"Block {i+1:02d}: {d_end:.1f} px ({change:+.1f})".replace('+', ''))

    avg_start = total_start_dist / len(blocks)
    avg_end = total_end_dist / len(blocks)
    avg_change = avg_end - avg_start
    sd_change = statistics.stdev(changes) if len(changes) > 1 else 0

    # Process Robot Stop Times
    stop_times = [r.stop_time for r in robots if r.stop_time is not None]
    if not stop_times:
        mean_stop, sd_stop = 0.0, 0.0
    else:
        mean_stop = statistics.mean(stop_times)
        sd_stop = statistics.stdev(stop_times) if len(stop_times) > 1 else 0

    screenshot_taken = False 

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False

        screen.fill(BG_COLOR)
        draw_text(screen, "SIMULATION COMPLETE", 32, 30, 20, (255, 215, 0))
        
        # COLUMN 1: Distance Stats
        draw_text(screen, f"Start AVG Dist: {avg_start:.1f} px", 18, 30, 70)
        draw_text(screen, f"End AVG Dist:   {avg_end:.1f} px", 18, 30, 95)
        sign_str = "" if avg_change < 0 else ""
        draw_text(screen, f"Net Change:     {sign_str}{avg_change:.1f} px", 18, 30, 120, (200, 200, 200))
        draw_text(screen, f"SD of Change:   {sd_change:.1f} px", 18, 30, 145, (200, 200, 200))

        # COLUMN 2: Time Stats
        draw_text(screen, f"Duration:       {duration:.2f} s", 18, 320, 70)
        draw_text(screen, f"Mean Stop Time: {mean_stop:.2f} s", 18, 320, 95)
        draw_text(screen, f"SD of Stops:    {sd_stop:.2f} s", 18, 320, 120, (200, 200, 200))
        
        if avg_change > 0:
            verdict = "AS EXPECTED: Pushed blocks away from entrance"
            v_color = REST_COLOR
        else:
            verdict = "NOT EXPECTED: Pushed blocks towards entrance"
            v_color = ROBOT_COLOR
        draw_text(screen, verdict, 20, 30, 185, v_color)

        draw_text(screen, "Individual Block Data:", 18, 30, 220, (150, 150, 150))
        
        y_start = 250
        line_height = 20
        # Display in 3 columns to save space
        for i in range(15):
            if i < len(block_stats):
                draw_text(screen, block_stats[i], 16, 30, y_start + (i * line_height))
        for i in range(15, min(30, len(block_stats))):
            draw_text(screen, block_stats[i], 16, 220, y_start + ((i-15) * line_height))
        for i in range(30, len(block_stats)):
            draw_text(screen, block_stats[i], 16, 410, y_start + ((i-30) * line_height))

        draw_text(screen, "Press [Close Window] to exit.", 16, 30, HEIGHT - 30, (100, 100, 100))

        pygame.display.flip()
        
        if not screenshot_taken:
            save_screenshot(screen, "04_final_metrics.png")
            export_metrics_to_csv(blocks, robots, duration, avg_start, avg_end, avg_change, sd_change, mean_stop, sd_stop)
            screenshot_taken = True
            
        pygame.time.Clock().tick(15)

# =============================================================================
# SECTION 5: MAIN EXECUTION
# =============================================================================
def main():
    print(f"Initializing Simulation")
    pygame.init()                
    pygame.font.init()           
    screen = pygame.display.set_mode((WIDTH, HEIGHT)) 
    pygame.display.set_caption(f"Braitenberg Vehicle Zero: Brownian Bots")
    clock = pygame.time.Clock()  

    robots = [Robot() for _ in range(NUM_ROBOTS)]
    
    blocks = []
    for i in range(NUM_BLOCKS):
        shape = "CIRCLE" if i < (NUM_BLOCKS // 2) else "SQUARE"
        blocks.append(Block(robots, shape))

    start_time = None
    final_time = None 
    
    last_screenshot_time = 0 
    start_screenshot_taken = False
    
    running = True            
    simulation_active = True  
    simulation_started = False 

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False
            if event.type == pygame.KEYDOWN:
                if not simulation_started:
                    if event.key == pygame.K_RETURN: 
                        simulation_started = True
                        start_time = time.time()
                        last_screenshot_time = start_time 
                else:
                    if event.key == pygame.K_RETURN: 
                        simulation_active = False 
                    if event.key == pygame.K_SPACE:  
                        main() 
                        return

        if simulation_active:
            if simulation_started:
                current_elapsed = time.time() - start_time
                
                check_collisions(robots, blocks)
                for b in blocks: b.update() 
                
                safe_robots = 0
                for r in robots:
                    r.sense(blocks, LIGHT_POS)   
                    r.decide_and_move()          
                    r.force_bounds()             
                    
                    if r.in_shadow:
                        safe_robots += 1
                        if r.stop_time is None:
                            r.stop_time = current_elapsed
                    else:
                        r.stop_time = None
                
                if safe_robots == NUM_ROBOTS and final_time is None:
                    final_time = current_elapsed
            
            screen.fill(BG_COLOR) 
            
            pygame.draw.circle(screen, (30, 30, 30), LIGHT_POS, MIN_SAFE_DISTANCE)
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, HEIGHT-WALL_THICKNESS, WIDTH, WALL_THICKNESS))
            pygame.draw.rect(screen, WALL_COLOR, (0, 0, WALL_THICKNESS, HEIGHT))
            pygame.draw.rect(screen, WALL_COLOR, (WIDTH-WALL_THICKNESS, 0, WALL_THICKNESS, HEIGHT))

            pygame.draw.circle(screen, LIGHT_COLOR, LIGHT_POS, LIGHT_RADIUS_VISUAL)
            
            if simulation_started:
                # Update live timer to show two decimal places
                time_to_show = final_time if final_time is not None else (time.time() - start_time)
                draw_text(screen, f"{time_to_show:.2f}s", 20, 10, 80, (255, 255, 0))

            for b in blocks: 
                if b.shape == "CIRCLE":
                    pygame.draw.circle(screen, BLOCK_COLOR, (int(b.x), int(b.y)), int(b.radius))
                else:
                    left = int(b.x - b.w/2)
                    top = int(b.y - b.h/2)
                    pygame.draw.rect(screen, BLOCK_COLOR, (left, top, b.w, b.h))
            
            for r in robots:
                color = REST_COLOR if r.in_shadow else ROBOT_COLOR
                pygame.draw.circle(screen, color, (int(r.x), int(r.y)), ROBOT_RADIUS)
                eye_x = r.x + math.cos(r.angle) * (ROBOT_RADIUS - 5)
                eye_y = r.y + math.sin(r.angle) * (ROBOT_RADIUS - 5)
                pygame.draw.circle(screen, (255, 255, 255), (int(eye_x), int(eye_y)), 3)

            if not simulation_started:
                draw_text(screen, "Press ENTER to Start", 32, WIDTH//2 - 120, 100, (255, 255, 0))
            else:
                draw_text(screen, f"Glare: 400px", 16, 20, HEIGHT - 50, (150, 150, 150))
                draw_text(screen, "Press ENTER for Metrics", 16, 20, HEIGHT - 30)

            pygame.display.flip()
            
            if simulation_started and not start_screenshot_taken:
                save_screenshot(screen, "01_sim_start.png")
                start_screenshot_taken = True
                
            if simulation_started:
                current_time = time.time()
                if current_time - last_screenshot_time >= 5:
                    timestamp = int(current_time - start_time)
                    save_screenshot(screen, f"02_progress_{timestamp}s.png")
                    last_screenshot_time = current_time

            clock.tick(60) 
        
        else:
            save_screenshot(screen, "03_sim_finished.png")
            
            duration_to_show = final_time if final_time is not None else (time.time() - start_time)
            show_results_screen(screen, blocks, robots, duration_to_show)
            running = False 

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()