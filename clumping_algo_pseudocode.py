#clumping_algo_pseudocode

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

    RETURN Best(population)CONSTANTS:
    POPULATION_SIZE = 20
    GENERATIONS = 20
    ELITISM_RATE = 0.2  (Top 20% survive)
    TRIALS = 5
    MUTATION_RATE = 0.2

FUNCTION Main():
    # 1. Initialization
    population = CreateRandomPopulation(POPULATION_SIZE)

    FOR generation FROM 1 TO GENERATIONS:
        scores = []

        # 2. Evaluation Phase
        FOR genome IN population:
            total_fitness = 0
            
            # Run multiple trials to account for randomness
            REPEAT TRIALS times:
                fitness = RunSimulation(genome)
                total_fitness = total_fitness + fitness
            
            average_fitness = total_fitness / TRIALS
            scores.append( (genome, average_fitness) )

        # 3. Selection Phase
        Sort scores by average_fitness DESCENDING
        survivors = Take Top(scores, count = POPULATION_SIZE * ELITISM_RATE)
        
        # 4. Reproduction Phase
        next_generation = copy(survivors)
        
        WHILE size(next_generation) < POPULATION_SIZE:
            parent = RandomChoice(survivors)
            child = Mutate(copy(parent))
            next_generation.append(child)

        population = next_generation

    RETURN Best(population)

FUNCTION RunSimulation(genome):
    Initialize Grid with scattered bricks
    Initialize Termites
    
    REPEAT 4000 steps:
        FOR each termite:
            Move termite randomly
            N = count_brick_neighbors(termite.location)
            
            IF termite is empty AND on brick:
                # Check Genes 0-4
                IF Random() < genome[N]: Pick up brick
            
            ELSE IF termite carrying AND on empty spot:
                # Check Genes 5-9
                IF Random() < genome[5 + N]: Drop brick

    RETURN CalculateClusteringMetric(Grid)

FUNCTION Mutate(genome):
    index = RandomInteger(0, 9)
    change = RandomFloat(-0.2, 0.2)
    genome[index] = Clamp(genome[index] + change, 0.0, 1.0)
    RETURN genome