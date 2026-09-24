# Titanic Survivor Prediction

This project uses C# and ML.NET to predict whether a Titanic passenger survived.

This model uses:

- Age
- Sex
- Pclass

to predict:

- Survived = 0 (Did not survive)
- Survived = 1 (Survived)

Since there are only two possible outcomes, this is a binary classification problem.

The data is split into 80% training data and 20% test data. The model learns from the training data and is evaluated using the test data.

## Logistic Regression

The project uses Logistic Regression because it is suitable for binary classification.

Instead of directly predicting 0 or 1, it calculates a probability between 0 and 1.

For example: probability 0.8 or 80 % chance of survived. This probability is then used to make the classification.

## Statistics

To evaluate the model, we first look at four possible results:

- TP (True Positive) predicted survived, and actually survived.
- TN (True Negative) predicted did not survive, and actually did not survive.
- FP (False Positive) predicted survived, but actually did not survive.
- FN (False Negative) predicted did not survive, but actually survived.

These values are used to calculate the metrics below.

### Accuracy

The percentage of all predictions that were correct.

Accuracy = (TP + TN) / (TP + TN + FP + FN)

### Presicion

Of the passengers predicted to have survived, how many actually survived.

Precision = TP / (TP + FP)

### Recall

Of the passengers who actually survived, how many did the model correctly identify.

Recall = TP / (TP + FN)

### F1 Score

A single metric that combines Precision and Recall. It is useful when we want to consider both false positives and false negatives.

F1 = 2 x (Presicion x Recall) / (Presicion + Recall)

## Why these metrics?

Each metric shows a different part of the model's performance

- Accuracy --- Overall correctness
- Precision --- How reliable positive predictions are
- Recall --- How many actual survivors were found
- F1 Score --- Balance between Precision and Recall
