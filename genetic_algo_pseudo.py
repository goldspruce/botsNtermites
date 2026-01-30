CONSTANTS:
    POPULATION_SIZE = 20
    GENERATIONS = 20
    SURVIVAL_RATE = 0.2 (Top 4 survive)
    MUTATION_STRENGTH = 0.2

FUNCTION Evolve():
    # 1. Initialize random population
    population = [RandomGenome() for i in 1 to POPULATION_SIZE]

    FOR generation in 1 to GENERATIONS:
        scores = []
        
        # 2. Assessment
        FOR genome in population:
            fitness = RunSimulation(genome) # Returns clustering score
            scores.append( (genome, fitness) )
        
        # 3. Selection (Survival of the Fittest)
        Sort scores by fitness DESCENDING
        survivors = Top 20% of scores
        
        # 4. Reproduction
        next_gen = []
        
        # Elitism: Keep the best unmodified
        FOR survivor in survivors:
            next_gen.append(survivor.genome)
            
        # Fill the rest with mutated clones
        WHILE len(next_gen) < POPULATION_SIZE:
            parent = RandomChoice(survivors)
            child = Clone(parent)
            
            # Mutation Step
            gene_index = RandomInteger(0, 9)
            mutation = RandomFloat(-0.2, 0.2)
            child.genes[gene_index] += mutation
            
            # Clamp constraints (probabilities must be 0.0 to 1.0)
            child.genes[gene_index] = Clamp(0.0, 1.0)
            
            next_gen.append(child)
            
        population = next_gen

    RETURN Best(population)