import pandas as pd
from scipy import stats

def load_data(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # Find the header row
    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("Trial,Duration (steps)"):
            header_idx = i
            break
    
    if header_idx == -1:
        return None
        
    # Read the rest as a dataframe
    df = pd.read_csv(file_name, skiprows=header_idx)
    return df['Mean Rock Displacement'].dropna().astype(float)

light_disp = load_data('aggregate_summary.csv')
dark_disp = load_data('dark_aggregate_summary.csv')

if light_disp is not None and dark_disp is not None:
    t_stat, p_val = stats.ttest_ind(light_disp, dark_disp, equal_var=False)
    print(f"Light samples: {len(light_disp)}")
    print(f"Dark samples: {len(dark_disp)}")
    print(f"Light Mean: {light_disp.mean():.4f}")
    print(f"Dark Mean: {dark_disp.mean():.4f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.4e}")
else:
    print("Could not find the header row in one or both files.")