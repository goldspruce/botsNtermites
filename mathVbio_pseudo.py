mathVbio_psudeo

import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(best_genome):
    # 1. The Theoretical Ideal (Deneubourg Model approximation)
    # Pick is high when neighbors are few (isolation).
    # Drop is high when neighbors are many (clustering).
    # k1 and k2 are sensitivity constants often set around 0.1 - 0.3
    k = 0.1 
    neighbors = np.array([0, 1, 2, 3, 4])
    
    # Deneubourg Formulas
    # P(pick) = (k / (k + neighbors))^2
    theo_pick = (k / (k + neighbors))**2
    
    # P(drop) = (neighbors / (k + neighbors))^2
    theo_drop = (neighbors / (k + neighbors))**2

    # 2. Your Evolved Genome
    evolved_pick = best_genome[0:5]
    evolved_drop = best_genome[5:10]

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Pick Probabilities
    ax1.plot(neighbors, theo_pick, 'k--', label='Theoretical (Deneubourg)', alpha=0.6)
    ax1.plot(neighbors, evolved_pick, 'ro-', label='AI Evolved', linewidth=2)
    ax1.set_title("Picking Strategy")
    ax1.set_xlabel("Number of Brick Neighbors")
    ax1.set_ylabel("Probability to Pick Up")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True)
    ax1.legend()

    # Plot Drop Probabilities
    ax2.plot(neighbors, theo_drop, 'k--', label='Theoretical (Deneubourg)', alpha=0.6)
    ax2.plot(neighbors, evolved_drop, 'bo-', label='AI Evolved', linewidth=2)
    ax2.set_title("Dropping Strategy")
    ax2.set_xlabel("Number of Brick Neighbors")
    ax2.set_ylabel("Probability to Drop")
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True)
    ax2.legend()

    plt.suptitle(f"Man vs. Machine: Optimization Strategy Comparison", fontsize=14)
    plt.tight_layout()
    plt.show()

# Run this if you have a best_genome_overall variable from your previous run
if 'best_genome_overall' in locals() and best_genome_overall is not None:
    plot_comparison(best_genome_overall)
else:
    print("No evolved genome found in memory. Run the evolution first!")