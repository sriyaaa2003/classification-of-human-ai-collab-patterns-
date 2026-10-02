# Classifying Human–AI Collaboration Patterns in Text

Six-class classification of how a text was produced, from fully human to fully AI-written, using interpretable stylometric features and a hierarchical classifier, plus association-rule mining to describe what distinguishes each class.

## Classes
| ID | Class |
|----|-------|
| 0 | Fully Human |
| 1 | Human-AI Polished |
| 2 | AI-AI Humanized |
| 3 | Human-AI Continued |
| 4 | Deeply Mixed |
| 5 | AI-Human Edited |

## Pipeline (run the notebooks in this order)
1. **`dataexploration.ipynb`**: loads the provided train/test JSONL files (288,918 + 72,661 articles) and inspects class balance and text length. The original split is skewed (class 3 is 3.7% of train but 51.2% of test).
2. **`dataprep.ipynb`**: merges everything (361,579 articles), makes a new stratified split (289,263 train / 72,316 test), and extracts 15 stylometric features with spaCy: burstiness, noun/verb/adjective/adverb ratios, POS diversity, syntactic complexity, pronouns, contractions, hapax legomena, formal transitions, passive voice, n-gram repetition, style breaks and lexical density.
3. **`hier.ipynb`**: hierarchical classifier (group detectors for classes 0–2 and 4–5, SMOTE for the rare class 5) compared with Gradient Boosting and Random Forest, then combined in a weighted-vote ensemble. The hierarchical model uses XGBoost classifiers.
4. **`binandpm.ipynb`**: discretises features into bins with decision-tree splits, mines association rules per class with a custom Apriori implementation, and turns them into 149 class-specific rules and readable feature profiles.
5. **`rules+hier.ipynb`**: blends the rule-based scores with the hierarchical model.

## Results (held-out 72,316 articles)
| Model | Accuracy | Macro F1 |
|-------|---------:|---------:|
| Hierarchical classifier | 62.15% | 0.4895 |
| Random Forest (tuned) | 59.28% | 0.5162 |
| Ensemble (hierarchical + Gradient Boosting + Random Forest) | 62.19% | 0.5445 |
| Hierarchical + mined rules | 62.14% | 0.4897 |

Mixed-authorship classes are hard: the two rarest classes score the lowest (class 5 F1 ≈ 0.07 for the hierarchical model). Adding the mined rules gave essentially no gain (+0.0002 macro F1), but the rules are useful for explanation, e.g. fully human text tends to show high burstiness and few formal transitions, while AI-AI humanized text shows low burstiness and high lexical density.

## Setup
Python 3 with `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `spacy` (with an English model), `matplotlib`, `seaborn`.

The notebooks expect the dataset in `../data/` (`train.jsonl`, `test.jsonl`) and write intermediate files to `../output/`. Neither folder is included in this repository.
