# AI for Non-Deterministic Board Games: Catan

This repository contains the code and research for my dissertation project on AI for non-deterministic board games, focusing on the game **Catan**. The project explores the design and performance of an AI model that can play Catan, with an emphasis on handling the game's non-deterministic elements and analyzing model strategies.

## Project Overview

1. **Mini-Catan Implementation**: Starting with a simplified version of Catan to establish the AI model's core mechanics and develop basic decision-making strategies.
2. **AI Model Development**: Creating an AI capable of playing Catan. The model will be trained and tested against known strategies, with a focus on adaptability and decision-making under uncertainty.
3. **Expansion to Full Catan**: Scaling the AI from mini-Catan to the complete game, incorporating the full set of Catan rules, board complexity, and strategies.
4. **Comparative Analysis**: The AI model will be benchmarked against previous models and techniques in Catan to assess performance, decision-making, and strategy efficacy.
5. **Rule Variation Experiments**: Experimenting with modifications to Catan’s rules and analyzing how these changes affect the AI's behavior and strategy.

## Repository Structure

- **`mini_catan/`**: Implementation of the simplified version of Catan and initial AI.
- **`full_catan/`**: Full Catan game code and AI implementation.
- **`analysis/`**: Code for comparing the model’s performance with other AI implementations.
- **`experiments/`**: Rule variation experiments and related analysis.

## Getting Started

### Prerequisites
- Python 3.x
- Required libraries can be installed via:
  ```bash
  pip install -r requirements.txt
  ```
### Running the Project
- To run the mini-Catan simulation:
  ```bash
  python mini_catan/main.py
  ```
- For the full Catan game:
  ```bash
  python full_catan/main.py
  ```
## Goals
- Effective AI for Non-Deterministic Gameplay: Design AI strategies that perform well despite the unpredictability inherent in Catan.
- Strategic Comparisons: Evaluate how the model measures up against established Catan-playing AIs.
- Impact of Rule Changes: Understand how AI adapts to new rules and what this reveals about strategy flexibility.
