import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load datasets
df_normal = pd.read_csv("features_normal.csv")
df_timing = pd.read_csv("features_timing_attack.csv")
df_replay = pd.read_csv("features_replay_attack.csv")
df_command = pd.read_csv("features_command_injection.csv")

# Combine
df = pd.concat([df_normal, df_timing, df_replay, df_command], ignore_index=True)

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)
model.fit(X_train, y_train)

importances = model.feature_importances_
for name, val in sorted(zip(X.columns, importances), key=lambda x: -x[1]):
    print(f"{name:25s}: {val:.3f}")

# Evaluate
y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
