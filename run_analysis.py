import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from agent import SemanticRatingAgent
agent = SemanticRatingAgent()
agent.run_full_analysis('data/Mobile Reviews Sentiment.csv', 'memory.json')
print('=== ANALYSIS DONE ===')
