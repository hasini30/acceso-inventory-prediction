## Setup and Execution

### 1. Install Dependencies

Clone the repository and move into the project folder:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd acceso-inventory-prediction
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Run the Project

**Train and evaluate the models**

```bash
python train_pipeline.py
```

This trains the candidate models, evaluates them, and saves the best one as `best_model.joblib`.

**Generate an inventory batch prediction**

```bash
python predict.py
```

This loads `best_model.joblib` and produces predictions for a batch of inventory. Run the training step first so the model file exists.

---

## Business Decision: False Positives vs. False Negatives

### Which error costs more?

| Error type | What happens | Business impact |
|---|---|---|
| **False Negative** | Inventory that should go to the kitchen is not flagged | Product stays on grocery shelves, expires, and is wasted |
| **False Positive** | Inventory is flagged that did not need transfer | Unnecessary kitchen transfers and extra operational cost |

For Accesco, a **False Negative is the more expensive mistake**. Wasted product is a direct loss, while an unneeded transfer is a smaller, recoverable cost.

### Threshold adjustment

To prioritize catching inventory at risk of expiring, the classification threshold was lowered from **0.50 to 0.35**.

| Metric (test set) | Threshold 0.50 | Threshold 0.35 | Change |
|---|---|---|---|
| Recall | 96.51% | 98.84% | +2.33 pts |
| Precision | 98.22% | 94.97% | -3.25 pts |
| False Negatives | 6 | 2 | -4 |
| False Positives | 3 | 9 | +6 |

### The trade-off

Lowering the threshold catches more at-risk inventory (4 fewer missed items) at the price of 6 additional unnecessary transfers. Because wastage is costlier than an extra transfer, this trade is favorable.

The right threshold ultimately depends on the real relative costs of **inventory wastage** and **unnecessary kitchen transfers**. If those costs change, the threshold should be revisited.