# Abstraction-and-Reasoning-Corpus---ARC
Hybrid neuro-symbolic solver for ARC-AGI-2: a neural-ranked DSL beam search with an LLM code-generation fallback, packaged and served with MLflow.
What is this?

ARC-AGI-2 tests whether a system can infer an abstract rule from a few input/output grid examples and apply it to a new grid. This repository is an experimental, research-style baseline for that problem. It combines symbolic program search with a small neural model and an LLM.

What I did in this project
Data loading: loaded the ARC-AGI-2 training (1000), evaluation (120) and test (240) tasks.
Object extraction: built background detection and connected-component object extraction with SciPy, shared by all primitives.
DSL of 135 primitives: rigid transforms, color remapping (90 pairs), tiling (up to 5x5), object cropping, symmetry completion, gravity in four directions and integer scaling.
Chain scoring: applied primitive chains to the training pairs and scored them by exact match.
Neural primitive ranker: trained a PyTorch MLP on 200 tasks to predict which primitives are likely to help, given simple grid features.
Guided beam search: searches chains up to depth 3 with beam width 25 and the 40 top-ranked primitives per step.
LLM fallback: for unsolved tasks, prompted openai/gpt-oss-20b through the Groq API to write a solve(grid) function. This includes retry logic, rate-limit handling and API-key rotation.
Combined solver and submission: DSL first, LLM code second. Generates a Kaggle-format submission.json with attempt_1 and attempt_2, plus a format validator.
MLflow deployment: packaged the ranker and search pipeline as a custom pyfunc model. It is logged with parameters and metrics, registered as ARC_Hybrid_Solver and served through mlflow models serve.
