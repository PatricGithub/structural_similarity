# Analysis of Structural Similarity in Chess Analogies using Large Language Models

**Authors:** Patric Pförtner and Penka Hristova

**Date:** July 31, 2025

---

## Project Description

This repository contains the data, analysis scripts, and prompts for the study on analogical reasoning in chess. The project investigates how chess players of varying skill levels explain their reasoning when constructing analogies.

We analyzed the written responses from chess players tasked with explaining their reasoning after creating an analogous board position. These explanations were prompted by the instruction, “Explain your analogy” (“Erklären Deine Analogie”). Four distinct large language models were employed to systematically evaluate these responses. The analytical prompts were grounded in Gentner's (1983) theory of structural similarity.

## Methodology

* **Data Collection:** Written explanations were collected from chess players immediately after they generated an analogous board position. The raw data can be found in the `/300_final.csv`.
* **Analysis Tools:** Four different large language models (LLMs) from the OPENROUTER model library were used to analyze the textual data. The models are specified within the analysis script.
* **Theoretical Framework:** The analysis is based on the definition of **structural similarity** from Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy. Cognitive Science, 7(2), 155–170. https://doi.org/10.1016/s0364-0213(83)80009-3. 

## Repository Contents
* `/structural_similarity_results`: Contains the raw and processed data from the study (`300_final.csv`).
* `/prompts.py`: Contains the exact prompts used to query the large language models.
* `/main.py`: Includes the Python analysis script.
* `README.md`: This file.

## How to Cite

If you use the code or data from this repository in your research, please cite it as follows:

Pförtner, P., & Hristova, P. (2025). *Analysis of Structural Similarity in Chess Analogies using Large Language Models* [Computer software]. https://github.com/PatricGithub/structural_similarity
