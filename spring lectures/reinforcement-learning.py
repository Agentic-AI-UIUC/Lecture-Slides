"""
DSPy Prompt Optimization Demo
==============================
Demonstrates how DSPy can automatically improve a prompt by optimizing
few-shot examples and instructions using BootstrapFewShotWithRandomSearch.

Task: Given a question, produce a short factual answer.
We use a small hand-crafted dataset, evaluate with exact-match,
and show before/after optimization scores.

Requirements:
    pip install dspy
"""

import os
import dspy
import random

# ── 1. Configure the language model ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Configuring the language model")
print("=" * 60)
print("  Model: gemini/gemini-2.5-flash (temperature=0.0)")
print("  This is the LLM that DSPy will call under the hood.\n")

os.environ["GEMINI_API_KEY"] = "AIzaSyBGFvMRoGJPv5RWKXhKt5EQ1QQ18JMYmBQ"
lm = dspy.LM("gemini/gemini-2.5-flash", temperature=0.0)
dspy.configure(lm=lm)

# ── 2. Build a small Q&A dataset (inline, no downloads needed) ───────────────
raw_data = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    {"question": "What planet is known as the Red Planet?", "answer": "Mars"},
    {"question": "What is the largest ocean on Earth?", "answer": "Pacific Ocean"},
    {"question": "What is the speed of light in km/s (approximate)?", "answer": "300000"},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"question": "What is the smallest prime number?", "answer": "2"},
    {"question": "What gas do plants absorb from the atmosphere?", "answer": "Carbon dioxide"},
    {"question": "What is the hardest natural substance on Earth?", "answer": "Diamond"},
    {"question": "In what year did the Titanic sink?", "answer": "1912"},
    {"question": "What is the powerhouse of the cell?", "answer": "Mitochondria"},
    {"question": "How many continents are there?", "answer": "7"},
    {"question": "What element does 'O' represent on the periodic table?", "answer": "Oxygen"},
    {"question": "Who developed the theory of general relativity?", "answer": "Albert Einstein"},
    {"question": "What is the largest mammal?", "answer": "Blue whale"},
    {"question": "What language has the most native speakers?", "answer": "Mandarin Chinese"},
    {"question": "What is the boiling point of water in Celsius?", "answer": "100"},
    {"question": "Who was the first person to walk on the Moon?", "answer": "Neil Armstrong"},
    {"question": "What is the currency of Japan?", "answer": "Yen"},
    {"question": "What is the tallest mountain in the world?", "answer": "Mount Everest"},
    {"question": "What organ pumps blood through the body?", "answer": "Heart"},
    {"question": "What is the square root of 144?", "answer": "12"},
    {"question": "Who discovered penicillin?", "answer": "Alexander Fleming"},
    {"question": "What is the chemical formula for water?", "answer": "H2O"},
    {"question": "What is the most abundant gas in Earth's atmosphere?", "answer": "Nitrogen"},
    {"question": "How many bones are in the adult human body?", "answer": "206"},
    {"question": "What country has the largest population?", "answer": "India"},
    {"question": "What year did World War II end?", "answer": "1945"},
    {"question": "What is the freezing point of water in Fahrenheit?", "answer": "32"},
]

# Convert to DSPy Examples and split into train / dev sets
print("=" * 60)
print("STEP 2: Preparing the dataset")
print("=" * 60)

examples = [dspy.Example(question=d["question"], answer=d["answer"]).with_inputs("question") for d in raw_data]
random.seed(42)
random.shuffle(examples)

trainset = examples[:20]  # used by the optimizer
devset = examples[20:]    # used for evaluation

print(f"  Total examples:  {len(examples)}")
print(f"  Training set:    {len(trainset)} (used by the optimizer to find good demos)")
print(f"  Dev/eval set:    {len(devset)} (held out to measure accuracy)")
print(f"  Sample question: \"{trainset[0].question}\"")
print(f"  Sample answer:   \"{trainset[0].answer}\"\n")


# ── 3. Define a metric ───────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Defining the evaluation metric")
print("=" * 60)
print("  Metric: case-insensitive substring match")
print("  A prediction is correct if the gold answer appears")
print("  anywhere in the model's response (e.g. 'Paris' in")
print("  'The capital of France is Paris.').\n")

def answer_match(example, pred, trace=None):
    """Check if the predicted answer contains the gold answer (case-insensitive)."""
    return example.answer.lower() in pred.answer.lower()


# ── 4. Define the module (unoptimized baseline) ─────────────────────────────
print("=" * 60)
print("STEP 4: Creating the baseline module (no optimization yet)")
print("=" * 60)
print("  Module: dspy.ChainOfThought(\"question -> answer\")")
print("  This sends a basic prompt with no few-shot examples")
print("  and no optimized instructions — just the signature.\n")

baseline = dspy.ChainOfThought("question -> answer")


# ── 5. Evaluate the baseline ────────────────────────────────────────────────
print("=" * 60)
print("STEP 5: Evaluating the BASELINE (before optimization)")
print("=" * 60)
print(f"  Running the unoptimized module on {len(devset)} dev questions...")
print("  This establishes our starting accuracy.\n")

evaluator = dspy.Evaluate(devset=devset, metric=answer_match, num_threads=4, display_progress=True)
baseline_result = evaluator(baseline)
baseline_score = float(baseline_result)
print(f"\n  >>> Baseline accuracy: {baseline_score:.1f}% <<<\n")


# ── 6. Optimize the prompt with BootstrapFewShotWithRandomSearch ─────────────
print("=" * 60)
print("STEP 6: Optimizing the prompt")
print("=" * 60)
print("  Optimizer: BootstrapFewShotWithRandomSearch")
print("  What it does:")
print("    1. Samples subsets of training examples as few-shot demos")
print("    2. Bootstraps chain-of-thought reasoning for each demo")
print("    3. Tries 6 candidate prompt variants with different demo combos")
print("    4. Picks the variant that scores best on the training set")
print()
print("  Settings:")
print("    - max_bootstrapped_demos = 3  (auto-generated CoT examples)")
print("    - max_labeled_demos     = 3  (gold examples from trainset)")
print("    - num_candidate_programs = 6  (prompt variants to try)")
print()
print("  This may take a minute — the optimizer is making many LLM calls...\n")

optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=answer_match,
    max_bootstrapped_demos=3,   # up to 3 auto-generated chain-of-thought demos
    max_labeled_demos=3,        # up to 3 labeled demos from trainset
    num_candidate_programs=6,   # try 6 candidate prompt variants
    num_threads=4,
)

optimized = optimizer.compile(baseline, trainset=trainset)
print("\n  Optimization complete!\n")


# ── 7. Evaluate the optimized version ───────────────────────────────────────
print("=" * 60)
print("STEP 7: Evaluating the OPTIMIZED module")
print("=" * 60)
print(f"  Running the optimized module on the same {len(devset)} dev questions...")
print("  The prompt now includes auto-selected few-shot demos.\n")

optimized_result = evaluator(optimized)
optimized_score = float(optimized_result)
print()
print("  ┌─────────────────────────────────────┐")
print(f"  │  Baseline accuracy:  {baseline_score:5.1f}%          │")
print(f"  │  Optimized accuracy: {optimized_score:5.1f}%          │")
print(f"  │  Improvement:        {optimized_score - baseline_score:+5.1f}%          │")
print("  └─────────────────────────────────────┘")
print()


# ── 8. Show a side-by-side example ──────────────────────────────────────────
print("=" * 60)
print("STEP 8: Side-by-side comparison on a single question")
print("=" * 60)

test_q = "What is the chemical formula for water?"
print(f"  Sending the same question to both modules to compare.\n")
print(f"  Question: \"{test_q}\"\n")

baseline_pred = baseline(question=test_q)
print(f"  Baseline answer:   {baseline_pred.answer}")

optimized_pred = optimized(question=test_q)
print(f"  Optimized answer:  {optimized_pred.answer}")
print(f"  Expected answer:   H2O\n")


# ── 9. Inspect the optimized prompt ─────────────────────────────────────────
print("=" * 60)
print("STEP 9: Inspecting the optimized prompt")
print("=" * 60)
print("  Below is the actual prompt DSPy constructed and sent to the LLM.")
print("  Notice the few-shot demos and chain-of-thought reasoning that")
print("  the optimizer automatically selected and inserted.\n")

# Show the last LM call to see the actual prompt DSPy constructed
dspy.inspect_history(n=1)

print("\n" + "=" * 60)
print("DEMO COMPLETE")
print("=" * 60)
print("  Key takeaway: DSPy automatically found better few-shot demos")
print("  and chain-of-thought examples that improved accuracy from")
print(f"  {baseline_score:.1f}% to {optimized_score:.1f}% — without any manual prompt engineering.")
print("=" * 60)
