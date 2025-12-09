import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load CSV file (must be in same folder as your code)
df = pd.read_csv("synthetic_drug_response.csv")

# Show dataset
print(df)

# Accuracy calculation
accuracy = accuracy_score(df["true_label"], df["predicted_label"])
print("Accuracy =", accuracy * 100, "%")

# Plot true vs predicted values
plt.plot(df["true_label"], label="True")
plt.plot(df["predicted_label"], label="Predicted")
plt.title("True vs Predicted")
plt.xlabel("Samples")
plt.ylabel("Values")
plt.legend()
plt.show()