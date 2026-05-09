import pandas as pd
import numpy as np
from scipy import stats

def analyze_termite_clustering(filename):
    # Load the simulation data
    df = pd.read_csv(filename)
    
    # Calculate the net change in environmental order (delta)
    # Assuming columns exist for Step 0 and Step 1000 clustering scores
    df['Clustering Delta'] = df['Clustering Score Step 1000'] - df['Clustering Score Step 0']
    
    # Extract the target data array
    active_deltas = df['Clustering Delta'].dropna().astype(float)
    
    # Descriptive Statistics
    n = len(active_deltas)
    mean_delta = active_deltas.mean()
    sd_delta = active_deltas.std(ddof=1) # ddof=1 for sample standard deviation
    
    # Inferential Statistics: One-Sample T-Test
    # Null hypothesis: The net change in clustering order is zero (control condition)
    null_mean = 0.0
    t_stat, p_val = stats.ttest_1samp(active_deltas, null_mean)
    
    # Print the formatted output
    print("ACTIVE TERMITES")
    print(f"Mean = {mean_delta:.4f}")
    print(f"SD = {sd_delta:.4f}")
    print(f"T-statistic = {t_stat:.4f}")
    print(f"P-value = {p_val:.4f}")
    
    if p_val < 0.05 and mean_delta > 0:
        print("Conclusion: strong net increase in structural order")
    else:
        print("Conclusion: no significant increase in structural order")

# Run the analysis on the 10,000 trial dataset
# analyze_termite_clustering('termite_trials_10k.csv')