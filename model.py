import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Define paths for the data and the model
DATA_PATH = os.path.join("dataset", "advertising_dataset.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "lasso_regression_model.pkl")

# Create the model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    # Load the dataset
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATA_PATH}. Please ensure the file exists.")
    exit()

# Separate features (X) and the target variable (y)
# We will use 'TV_Ad_Budget', 'Radio_Ad_Budget', and 'Newspaper_Ad_Budget' to predict 'Sales'.
features = ['TV_Ad_Budget', 'Radio_Ad_Budget', 'Newspaper_Ad_Budget']
target = 'Sales'

X = df[features]
y = df[target]

# Split the data into training and testing sets
# The test size is 20% of the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Lasso regression is sensitive to the scale of the features.
# We will use a StandardScaler to normalize the data.
# The scaler is trained on the training data and then used to transform both sets.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Lasso Regression model with a regularization parameter (alpha).
# The alpha parameter controls the strength of the penalty. A larger alpha
# increases the penalty, which is more likely to shrink coefficients to zero.
model = Lasso(alpha=1.0)

# Train the model using the scaled training data
print("Training the Lasso Regression model...")
model.fit(X_train_scaled, y_train)
print("Training complete.")

# Save the trained model and the scaler to a pickle file.
# The scaler is saved along with the model to ensure new data is preprocessed
# consistently before making predictions with the saved model.
with open(MODEL_PATH, "wb") as f:
    # We save both the scaler and the model in a dictionary for easy loading.
    pickle.dump({'model': model, 'scaler': scaler}, f)

print(f"Model and scaler successfully saved to {MODEL_PATH}")
