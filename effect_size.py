import math

def calculate_one_sample_cohens_d(sample_mean, sample_sd, null_mean=0.0):
    """
    Calculates Cohen's d for a one-sample t-test.
    
    Formula:
        d = (Sample Mean - Null Hypothesis Mean) / Sample Standard Deviation
        
    Interpretation Guidelines (Cohen, 1988):
        0.2 = Small effect
        0.5 = Medium effect
        0.8 = Large effect
    """
    # Calculate the absolute difference between the observed mean and the baseline null
    mean_difference = sample_mean - null_mean
    
    # Standardize the difference by dividing by the standard deviation
    d_value = mean_difference / sample_sd
    
    return d_value


def calculate_two_sample_cohens_d(mean1, sd1, n1, mean2, sd2, n2, provided_pooled_sd=None):
    """
    Calculates Cohen's d for an independent two-sample t-test.
    
    If the pooled standard deviation is already known/provided, it uses that directly.
    Otherwise, it computes the pooled standard deviation using the sample sizes (n)
    and individual standard deviations.
    
    Formula for Pooled SD:
        SD_pooled = sqrt( ((n1 - 1)*sd1^2 + (n2 - 1)*sd2^2) / (n1 + n2 - 2) )
        
    Formula for Cohen's d:
        d = (Mean1 - Mean2) / SD_pooled
    """
    # Step 1: Establish the standard deviation denominator
    if provided_pooled_sd is not None:
        # If the statistical software already outputted a pooled SD, use it directly
        pooled_sd = provided_pooled_sd
    else:
        # Compute the pooled variance manually using degrees of freedom weighting
        degrees_of_freedom = n1 + n2 - 2
        pooled_variance = (((n1 - 1) * (sd1 ** 2)) + ((n2 - 1) * (sd2 ** 2))) / degrees_of_freedom
        pooled_sd = math.sqrt(pooled_variance)
    
    # Step 2: Calculate the raw separation between the two group means
    mean_difference = mean1 - mean2
    
    # Step 3: Standardize the effect size
    d_value = mean_difference / pooled_sd
    
    return d_value, pooled_sd


def interpret_cohens_d(d):
    """
    Returns a qualitative interpretation of Cohen's d magnitude threshold.
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible / Microscopic"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    elif abs_d < 1.2:
        return "Large"
    else:
        return "Very Large (Profound)"


# =============================================================================
# VERIFICATION EXECUTION (Using Simulation Results Data)
# =============================================================================
if __name__ == "__main__":
    print("--- EFFECT SIZE ANALYSIS REPORT ---\n")
    
    sample_size = 10000  # N = 10,000 for all tracking scenarios
    
    # -------------------------------------------------------------------------
    # Case 1: One-Sample Check (BBOT LIGHT vs. Theoretical Randomness of 0)
    # ------------------------------------------------#
    bbot_light_mean = 35.5300
    bbot_light_sd = 50.0778
    
    d_bbot_one_sample = calculate_one_sample_cohens_d(bbot_light_mean, bbot_light_sd, null_mean=0.0)
    print(f"1. One-Sample BBot Light vs. Null (0):")
    print(f"   Cohen's d = {d_bbot_one_sample:.4f} ({interpret_cohens_d(d_bbot_one_sample)})\n")
    
    # -------------------------------------------------------------------------
    # Case 2: Independent Two-Sample Check (BBOT LIGHT vs. BBOT DARK)
    # -------------------------------------------------------------------------
    bbot_dark_mean = -0.1560
    bbot_dark_sd = 30.0456
    bbot_pooled_sd_provided = 41.2946 # From Welch/pooled T-test output
    
    # Calculate using the provided pooled SD
    d_bbot_two_sample, _ = calculate_two_sample_cohens_d(
        mean1=bbot_light_mean, sd1=bbot_light_sd, n1=sample_size,
        mean2=bbot_dark_mean,  sd2=bbot_dark_sd,  n2=sample_size,
        provided_pooled_sd=bbot_pooled_sd_provided
    )
    
    print(f"2. Two-Sample BBot (Light vs. Dark Environment):")
    print(f"   Mean Difference = {bbot_light_mean - bbot_dark_mean:.4f} px")
    print(f"   Cohen's d       = {d_bbot_two_sample:.4f} ({interpret_cohens_d(d_bbot_two_sample)})\n")

    # -------------------------------------------------------------------------
    # Case 3: Independent Two-Sample Check (ROCK LIGHT vs. ROCK DARK)
    # -------------------------------------------------------------------------
    rock_light_mean = 42.4717
    rock_light_sd = 24.7422
    
    rock_dark_mean = 8.4181
    rock_dark_sd = 31.0622
    rock_pooled_sd_provided = 28.006
    
    # Calculate using provided pooled SD
    d_rock_two_sample, calculated_pooled_sd = calculate_two_sample_cohens_d(
        mean1=rock_light_mean, sd1=rock_light_sd, n1=sample_size,
        mean2=rock_dark_mean,  sd2=rock_dark_sd,  n2=sample_size,
        provided_pooled_sd=rock_pooled_sd_provided
    )
    
    # Verify what it would be if calculated automatically from standard deviations
    _, auto_pooled_sd = calculate_two_sample_cohens_d(
        mean1=rock_light_mean, sd1=rock_light_sd, n1=sample_size,
        mean2=rock_dark_mean,  sd2=rock_dark_sd,  n2=sample_size,
        provided_pooled_sd=None # Force automated math computation
    )
    
    print(f"3. Two-Sample Passive Rocks (Light vs. Dark Environment Noise):")
    print(f"   Mean Difference    = {rock_light_mean - rock_dark_mean:.4f} px")
    print(f"   Calculated Pool SD = {auto_pooled_sd:.4f} px (Automated formula derivation)")
    print(f"   Cohen's d          = {d_rock_two_sample:.4f} ({interpret_cohens_d(d_rock_two_sample)})\n")
    print("-----------------------------------------------------------------")