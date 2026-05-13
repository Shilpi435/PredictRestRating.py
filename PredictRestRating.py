import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Create sample restaurant dataset
data = {
    "Price": [500, 1000, 800, 1200, 600, 1500, 700, 900],
    "Votes": [120, 300, 250, 400, 180, 500, 200, 350],
    "OnlineDelivery": ["Yes", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
    "TableBooking": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
    "Aggregate rating": [3.5, 4.5, 3.8, 4.8, 3.2, 4.9, 3.6, 4.4]
}

# Convert into dataframe
df = pd.DataFrame(data)

# Encode categorical columns
le = LabelEncoder()

df["OnlineDelivery"] = le.fit_transform(df["OnlineDelivery"])
df["TableBooking"] = le.fit_transform(df["TableBooking"])

# Features and target
X = df.drop("Aggregate rating", axis=1)
y = df["Aggregate rating"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("Predicted Ratings:", y_pred)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))