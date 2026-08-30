import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from agent import SemanticRatingAgent, SemanticReasoningEngine

print("=" * 62)
print("VERIFICATION SUITE: UNSEEN REVIEW VALIDATION & DATASET Q&A")
print("=" * 62)

agent = SemanticRatingAgent()
agent.load_memory("memory.json")
agent.reasoning_engine = SemanticReasoningEngine()

sample_unseen_reviews = [
    "Loving the clean UI and fast updates. Best purchase of the year!",
    "Battery drains completely in just 4 hours and charging is very slow. Returning it.",
    "Decent phone for casual daily use, nothing extraordinary but works fine.",
    "The camera is great in daylight but night shots are terrible. Mixed feelings.",
]

print("\n" + "#" * 62)
print("TESTING UNSEEN REVIEWS (VALIDATION MODE)")
print("#" * 62)

for rev in sample_unseen_reviews:
    print(f"\nYou: {rev}")
    print("\nAgent:")
    print(agent.answer_question(rev))
    print("-" * 62)

print("\n" + "#" * 62)
print("TESTING DATASET QUESTION")
print("#" * 62)

q = "Which variables are most strongly associated with the overall rating?"
print(f"\nYou: {q}")
print("\nAgent:")
print(agent.answer_question(q))
print("=" * 62)
