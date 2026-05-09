import pandas as pd
from scipy import stats

filename = "termite_blind_net_change_data_2026-04-30_17-17-02.csv"
df = pd.read_csv(filename)

n_trials = len(df)
mean_delta = df['Delta_Score'].mean()
sd_delta = df['Delta_Score'].std()

t_stat, p_val = stats.ttest_1samp(df['Delta_Score'], 0.0)

print(f"N: {n_trials}")
print(f"Mean Delta: {mean_delta}")
print(f"SD Delta: {sd_delta}")
print(f"T-stat: {t_stat}")
print(f"P-value: {p_val}")