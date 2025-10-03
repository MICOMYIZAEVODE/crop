import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
data = pd.read_excel("Crop_recommendation.xlsx")
print(data)

X = data.drop("crop", axis=1)
y = data["crop"]

# 2. Split data
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", C=1, gamma="scale"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(random_state=42)
}

results = {}

print("Training models...")
# 4. Train & evaluate
for name, clf in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=5)),   # adjust components if needed
        ("clf", clf)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = (acc, pipeline)
    print(f"{name} Accuracy: {acc:.4f}")

# 5. Pick best model
best_model_name = max(results, key=lambda x: results[x][0])
best_model = results[best_model_name][1]

print(f"\n✅ Best Model: {best_model_name} with Accuracy = {results[best_model_name][0]:.4f}")

# 6. Save model
joblib.dump(best_model, "crop_model.joblib")
print("Model saved as crop_model.joblib")
