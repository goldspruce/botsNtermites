# 1. Define your Personas (The "Crowd")
personas = [
    "A busy soccer mom who relies on Facebook for news",
    "A tenured professor of geography",
    "A high school student who guesses based on intuition",
    "A conspiracy theorist who mistrusts official narratives",
    # ... generate 20-50 of these
]

# 2. Define the Question
question = "Is Philadelphia the capital of Pennsylvania? (A: Yes, B: No)"

# 3. The Loop
results = []
for p in personas:
    prompt = f"You are {p}... {question}..."
    response = call_llm(prompt) # Returns the JSON format
    results.append(response)

# 4. The Math (Bayesian Truth Serum)
# Calculate Actual Vote %
vote_counts = count_votes(results) 
actual_A_percent = vote_counts['A'] / total_personas

# Calculate Predicted Vote %
# Average the "prediction_A" field from ALL responses
avg_predicted_A = mean([r['prediction_A'] for r in results])

# 5. The Winner
ratio_A = actual_A_percent / avg_predicted_A
if ratio_A > 1:
    print("Surprisingly Popular Answer: A")
else:
    print("Surprisingly Popular Answer: B")