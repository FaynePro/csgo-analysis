# Tactic Classifier

Run the classifier:

```bash
python src/ml/tactic_classifier/classifier.py
```

## Modes

### Single Train–Test Split (default)

Runs when no additional arguments are provided.

- Performs a single train–test split  
- Trains the model  
- Prints results to the console  
- Does not export confusion matrix or metrics to the output directory  

### K-Fold Cross-Validation

```bash
python src/ml/tactic_classifier/classifier.py --kfold
```

- Runs k-fold cross-validation  
- Exports metrics (e.g., confusion matrix) to the output directory  

## Input

The model expects as input the raw data exported from the `data_preprocessor` class.

## Current Limitations

- It is not yet possible to use the model directly on raw data for prediction.  
- Prediction functions are work in progress.
