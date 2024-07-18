from xgboost import XGBClassifier

from data_loader import load_data

# XGBoost model
# Adapted from https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/

(train_data, val_data, test_data), (train_targets, val_targets, test_targets) = load_data()

model = XGBClassifier()
model.fit(train_data, train_targets)

predictions = model.predict(test_data)
correct = (predictions.argmax(1) == test_targets.argmax(1)).sum() / predictions.shape[0]
print(f"Accuracy: {correct*100:.1f}%" )
