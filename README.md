# Anthropic ML Technical Exercise

## Author
Yash - AI/ML Engineer

## Overview
This repository contains my submission for the Anthropic ML Technical Exercise, demonstrating:
- **Part 1**: Transformer-based sentiment analysis with attention visualization
- **Part 2**: Research analysis on conversational AI safety and reliability
- **Part 3**: Code review and optimization of transformer implementation
- **Bonus**: Scaling law analysis and predictions

## Repository Structure
```
anthropic-ml-exercise/
├── README.md
├── requirements.txt
├── part1_implementation/
│   ├── model.py              # Custom transformer encoder with attention
│   ├── train.py              # Training loop with logging & checkpoints
│   ├── evaluate.py           # Evaluation metrics and error analysis
│   ├── config.yaml           # Hyperparameter configuration
│   ├── data_utils.py         # Data loading and preprocessing
│   ├── attention_viz.py      # Attention visualization utilities
│   └── report.md             # Summary of approach and findings
├── part2_research_analysis/
│   └── technical_document.md # AI safety research analysis
├── part3_code_review/
│   ├── review.md             # Detailed code review
│   └── improved_code.py      # Refactored implementation
└── bonus_scaling_laws/
    └── scaling_analysis.ipynb # Scaling law analysis notebook
```

## Quick Start

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Part 1: Model Training
```bash
cd part1_implementation

# Train the model
python train.py --config config.yaml

# Evaluate on test set
python evaluate.py --checkpoint checkpoints/best_model.pt

# Generate attention visualizations
python attention_viz.py --checkpoint checkpoints/best_model.pt --text "This movie was amazing!"
```

### Reproducing Results
- **Random Seeds**: All experiments use seed=42 for reproducibility
- **Hardware**: Tested on NVIDIA RTX 3090 (24GB VRAM), also runs on CPU
- **Expected Training Time**: GPU ~30 minutes, CPU ~5 hours

### Key Results (Part 1)
| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | 88.2% | 87.6% |
| F1 Score | 0.881 | 0.874 |
| AUC-ROC | 0.943 | 0.938 |

## License
MIT License
