# Digital Persona-Based Synthetic Survey Dataset (N=400)

## Overview
This dataset contains **400 unique synthetic consumer survey responses** generated across four empirically derived digital personas ($N=100$ per persona) for research in **Semantic Scale Ratings (SSR)** and **UTAUT2 technology adoption modeling**.

## Experimental Parameters
- **Generative Architecture:** Local Qwen 2.5:7B Instruct via Ollama local runtime (`http://localhost:11434`).
- **Prompt Register:** Natural, everyday Indian English.
- **Rating Constraint:** Textual natural language only. Zero numerical Likert ratings, scores, or star ratings.
- **Constructs Evaluated:**
  1. **PE:** Performance Expectancy (68 responses)
  2. **EE:** Effort Expectancy (68 responses)
  3. **HM:** Hedonic Motivation (65 responses)
  4. **SI:** Social Influence (75 responses)
  5. **PP:** Price / Perceived Value (57 responses)
  6. **BI:** Behavioral Intention (67 responses)

## Persona Distribution
- **P1: Premium Tier Owner ($N=100$):** High technology comfort, prefers premium quality, performance, reliability, and long-term utility.
- **P2: Mid-Range Heavy User ($N=100$):** Frequent technology user, values speed, multitasking, battery life, and stable performance.
- **P3: Budget Practical Adopter ($N=100$):** Practical user, prioritizes affordability, essential features, reliability, and everyday utility.
- **P4: Student Value Seeker ($N=100$):** Budget-constrained student, price-sensitive, prioritizes affordability, educational usefulness, and essential features.

## Files
- `synthetic_responses.csv`: The complete, verified 400-response dataset.
- `generation_metadata.json`: Machine-readable experiment parameters, timestamps, and model configs.
- `generation_log.csv`: Batch execution history and tracking.
- `README.md`: This documentation.

## Schema
`Sample_ID, Persona_ID, Persona_Name, Construct, Response_Text`
