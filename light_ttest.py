# 1. IMPORTING LIBRARIES
# pandas is the ultimate data manipulation library in Python. We use it to read and clean spreadsheets.
import pandas as pd

# scipy.stats contains all the heavy-duty statistical formulas (like t-tests, ANOVAs, etc.)
from scipy import stats

def analyze_displacements(filename):
    # =========================================================================
    # STEP 1: FIND WHERE THE ACTUAL SPREADSHEET STARTS
    # =========================================================================
    # We open the file in "read" mode ("r") and read all the lines of text.
    with open(filename, "r") as file:
        lines = file.readlines()
        
    header_idx = -1 # This will store the line number where the actual table headers are.
    
    # We loop through just the first 50 lines of the file.
    # enumerate() gives us both the line number (i) and the text of the line.
    for i, line in enumerate(lines[:50]):
        # We are looking for the row that contains the column names "Trial" and "Duration".
        if "Trial,Duration" in line:
            header_idx = i # We found it! Save the row number.
            break          # Stop looking, we don't need to check the rest of the lines.
            
    # If we couldn't find the header, stop the program and print an error.
    if header_idx == -1:
        print("Error: Could not find the data header.")
        return

    # =========================================================================
    # STEP 2: LOAD AND CLEAN THE DATA
    # =========================================================================
    # Now we tell pandas to read the CSV file, but we tell it to skip all the 
    # messy text rows at the top by using 'skiprows = header_idx'.
    df = pd.read_csv(filename, skiprows=header_idx)
    
    # We extract the specific column we want: 'Mean Rock Displacement'.
    # pd.to_numeric makes sure Python treats these as numbers, not text.
    # errors='coerce' tells Python: "If you find a weird word instead of a number, turn it into a blank (NaN)."
    # .dropna() removes any of those blank rows so they don't break our math.
    rock_means = pd.to_numeric(df['Mean Rock Displacement'], errors='coerce').dropna()
    
    # We do the exact same thing for the BBot displacement column.
    bbot_means = pd.to_numeric(df['Mean BBot Displacement'], errors='coerce').dropna()
    
    print(f"Successfully loaded {len(rock_means)} data points for Rocks.")
    
    # =========================================================================
    # STEP 3: RUN THE STATISTICAL TESTS (ONE-SAMPLE T-TEST)
    # =========================================================================
    # A one-sample t-test compares a group of numbers against a hypothetical "population mean".
    # We want to know if our displacements are significantly different from exactly zero.
    # popmean=0 sets our Null Hypothesis to exactly 0.
    t_rock, p_rock = stats.ttest_1samp(rock_means, popmean=0)
    t_bbot, p_bbot = stats.ttest_1samp(bbot_means, popmean=0)
    
    # =========================================================================
    # STEP 4: PRINT THE RESULTS FORMATTED NICELY
    # =========================================================================
    print("\n--- Rock Displacement vs 0 ---")
    print(f"Mean: {rock_means.mean():.4f} px")        # :.4f rounds the number to 4 decimal places
    print(f"SD: {rock_means.std():.4f} px")
    print(f"t-statistic: {t_rock:.4f}")
    print(f"p-value: {p_rock:.4e}")                 # :.4e prints very small numbers in scientific notation

    print("\n--- BBot Displacement vs 0 ---")
    print(f"Mean: {bbot_means.mean():.4f} px")
    print(f"SD: {bbot_means.std():.4f} px")
    print(f"t-statistic: {t_bbot:.4f}")
    print(f"p-value: {p_bbot:.4e}")

# Finally, we run the function on your specific file.
analyze_displacements("aggregate_summary.csv")