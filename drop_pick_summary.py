# Calculate Density (0.0 to 1.0)
density = n_cnt / 4.0

# RULE A: PICK UP
if not t.carrying and here == 1:
    # High chance to pick up if density is LOW (1.0 - density)
    if random.random() < (1.0 - density):
        t.carrying = True
        grid[t.x, t.y] = 0

# RULE B: DROP
elif t.carrying and here == 0:
    # High chance to drop if density is HIGH (density + small bias)
    if random.random() < (density + 0.01):
        t.carrying = False
        grid[t.x, t.y] = 1