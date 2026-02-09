import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("diabetes.csv")

# Split into input (X) and output (y)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Optional: Predict for custom input
sample = [5, 116, 74, 0, 0, 25.6, 0.201, 30]  # Example input
sample_df = pd.DataFrame([sample], columns=X.columns)
prediction = model.predict(sample_df)

print("Diabetes Prediction:", "Positive" if prediction[0] == 1 else "Negative")
