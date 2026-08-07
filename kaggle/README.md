# Kaggle Projects

Machine learning competitions and experiments hosted on [Kaggle](https://www.kaggle.com/).

## Projects

| Project | Competition | Status |
|---------|-------------|--------|
| [rsna-knee-abnormality-detection](./rsna-knee-abnormality-detection/) | [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) | Active |

## Layout

Each project follows the same structure:

```
kaggle/<project-name>/
├── README.md          # Project overview and setup
├── config.py          # Paths, labels, and constants
├── requirements.txt   # Project-specific dependencies
├── data/              # Downloaded competition data (gitignored)
├── models/            # Saved checkpoints (gitignored)
├── submissions/       # Submission CSVs
├── notebooks/         # Exploration and Kaggle notebooks
└── src/               # Reusable Python modules
```
