import matplotlib.pyplot as plt
import numpy as np

# --- 1. THE DATA ---
# Replace these values with the results from your LLM experiment.
# We are plotting the stats for the CORRECT Answer (the "Minority Truth").
questions = [
    "Q1: Seasons\n(Axial Tilt)", 
    "Q2: Blood\n(Dark Red)", 
    "Q3: Napoleon\n(Average Ht)", 
    "Q4: Great Wall\n(Not Visible)", 
    "Q5: Frankenstein\n(The Creature)"
]

# The Actual Vote % the correct answer received (e.g., 30%)
actual_votes = [30, 40, 35, 25, 45] 

# The Predicted Vote % (What the crowd THOUGHT the correct answer would get)
# Ideally, this should be LOWER than the actual vote.
predicted_votes = [20, 10, 15, 5, 25]

# --- 2. THE PLOT SETUP ---
x = np.arange(len(questions))  # Label locations
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

# Plotting the bars
# Green for Actual (Truth), Gray for Prediction (Expectation)
rects1 = ax.bar(x - width/2, actual_votes, width, label='Actual Vote %', color='#4CAF50')
rects2 = ax.bar(x + width/2, predicted_votes, width, label='Predicted Vote %', color='#9E9E9E')

# --- 3. STYLING & LABELS ---
ax.set_ylabel('Percentage of Votes')
ax.set_title('Bayesian Truth Serum: Actual vs. Predicted (Correct Answers)')
ax.set_xticks(x)
ax.set_xticklabels(questions)
ax.legend()
ax.set_ylim(0, 100) # Keep scale consistent 0-100%

# --- 4. HIGHLIGHTING THE "SURPRISE" ---
# This loop adds the "+10%" label above the bars to show the magnitude of the surprise
for i in range(len(questions)):
    gap = actual_votes[i] - predicted_votes[i]
    if gap > 0:
        label = f"+{gap}% SURPRISE"
        color = 'green'
    else:
        label = f"{gap}% FAIL"
        color = 'red'
        
    ax.annotate(label,
                xy=(x[i], max(actual_votes[i], predicted_votes[i])), 
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontweight='bold', color=color)

plt.tight_layout()
plt.show()      