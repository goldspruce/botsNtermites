# 1. Create a population of random "Brain Parameters"
population = []
for i in range(20):
    # A brain is just a list of 10 probabilities (0% to 100%)
    brain = [random_percent() for _ in range(10)]
    population.append(brain)

# 2. The Cycle of Life (Generations)
for generation in range(15):
    
    # A. Test every brain (The "Exam")
    scores = []
    for brain in population:
        # Run a quick simulation using this specific brain's probabilities
        final_score = simulate_behavior(brain)
        scores.append( (final_score, brain) )
    
    # B. Survival of the Fittest
    scores.sort(best_to_worst)
    survivors = scores[0:4] # Keep top 4 (Top 20%)
    
    # C. Mutate and Reproduce
    next_generation = survivors.copy()
    
    while len(next_generation) < 20:
        # Pick a successful parent
        parent = random.choice(survivors)
        # Clone it
        child = parent.copy()
        # Mutate: Tweak one probability slightly (e.g., change 0.50 to 0.58)
        child = mutate(child) 
        next_generation.append(child)
        
    population = next_generation

# 3. Return the "Champion" brain to be visualized
return population[0]