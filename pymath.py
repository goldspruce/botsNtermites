import pandas as pd
import mpmath
from scipy import stats

# Set math library to 50 decimal places of precision
mpmath.mp.dps = 50

def exact_t_pvalue(t_val, v):
    """Calculates the exact 2-tailed p-value for extreme t-statistics."""
    t_mp = mpmath.mpf(str(t_val))
    dof_mp = mpmath.mpf(str(v))
    
    # 2-tailed p-value via the regularized incomplete beta function
    x = dof_mp / (dof_mp + t_mp**2)
    p = mpmath.betainc(dof_mp/2, mpmath.mpf('0.5'), 0, x, regularized=True)
    return p

def format_extreme_p(val):
    """Formats an arbitrary-precision float into scientific notation."""
    log10_val = mpmath.log10(val)
    exponent = int(mpmath.floor(log10_val))
    mantissa = float(val / (mpmath.power(10, exponent)))
    return f"{mantissa:.6f}e{exponent}"

# Load the Data
df = pd.read_csv("aggregate_summary.csv", skiprows=31) # Adjust skiprows if needed

rock_disp = pd.to_numeric(df['Mean Rock Displacement'], errors='coerce').dropna()
bbot_disp = pd.to_numeric(df['Mean BBot Displacement'], errors='coerce').dropna()

# 1. Run standard t-tests to get the t-statistics
t_rock, _ = stats.ttest_1samp(rock_disp, popmean=0)
t_bbot, _ = stats.ttest_1samp(bbot_disp, popmean=0)

dof_rock = len(rock_disp) - 1
dof_bbot = len(bbot_disp) - 1

# 2. Feed the t-statistics into mpmath for the exact p-values
exact_p_rock = exact_t_pvalue(t_rock, dof_rock)
exact_p_bbot = exact_t_pvalue(t_bbot, dof_bbot)

print(f"--- ROCK DISPLACEMENT ---")
print(f"t-statistic: {t_rock:.4f}")
print(f"p-value: {format_extreme_p(exact_p_rock)}")

print(f"\n--- BBOT DISPLACEMENT ---")
print(f"t-statistic: {t_bbot:.4f}")
print(f"p-value: {format_extreme_p(exact_p_bbot)}")