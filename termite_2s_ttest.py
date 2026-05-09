import pandas as pd
import numpy as np
from scipy import stats

def analyze_termite_two_sample(active_filename, control_filename):
    # 1. Load the simulation data for both conditions
    df_active = pd.read_csv(active_filename)
    df_control = pd.read_csv(control_filename)
    
    # 2. Calculate the net change in environmental order (delta) for Active
    # Assuming columns exist for Step 0 and Step 1000 clustering scores
    df_active['Clustering Delta'] = df_active['Clustering Score Step 1000'] - df_active['Clustering Score Step 0']
    active_deltas = df_active['Clustering Delta'].dropna().astype(float)
    
    # 3. Calculate the net change in environmental order (delta) for Control
    df_control['Clustering Delta'] = df_control['Clustering Score Step 1000'] - df_control['Clustering Score Step 0']
    control_deltas = df_control['Clustering Delta'].dropna().astype(float)
    
    # 4. Descriptive Statistics
    mean_active = active_deltas.mean()
    sd_active = active_deltas.std(ddof=1)
    
    mean_control = control_deltas.mean()
    sd_control = control_deltas.std(ddof=1)
    
    # 5. Inferential Statistics: Two-Sample T-Test 
    # equal_var=False applies Welch's correction for unequal variances
    t_stat, p_val = stats.ttest_ind(active_deltas, control_deltas, equal_var=False)
    
    # 6. Print the formatted output
    print("--- ACTIVE TERMITES VS CONTROL (BLINDED) ---")
    print(f"Active Mean  = {mean_active:.4f} (SD = {sd_active:.4f})")
    print(f"Control Mean = {mean_control:.4f} (SD = {sd_control:.4f})")
    print("-" * 44)
    print(f"T-statistic  = {t_stat:.4f}")
    print(f"P-value      = {p_val:.4f}")
    
    # 7. Formulate Conclusion
    if p_val < 0.05 and mean_active > mean_control:
        print("Conclusion: Strong net increase in structural order compared to the random control.")
    elif p_val < 0.05 and mean_active < mean_control:
        print("Conclusion: Significant DECREASE in structural order compared to the random control.")
    else:
        print("Conclusion: No significant difference in structural order between active and control groups.")

# Run the analysis on your datasets by un-commenting the line below:
# analyze_termite_two_sample('active_trials_10k.csv', 'control_trials_10k.csv')