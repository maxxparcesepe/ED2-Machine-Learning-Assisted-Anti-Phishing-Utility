# Import necessary libraries
import pandas as pd                     # For data manipulation and CSV handling
import matplotlib.pyplot as plt         # For plotting feature importance
import seaborn as sns                   # For styled plots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample      # For class balancing
import joblib                           # For saving the model and files

# STEP 1: Load Dataset 1 (Phishing_Legitimate_full.csv)
# -----------------------------------------------------
# This dataset includes both phishing and legitimate samples with 49+ features.
df1 = pd.read_csv("Phishing_Legitimate_full.csv")
df1 = df1.select_dtypes(include=["number"])            # Drop non-numeric columns (like URLs or filenames)
df1.rename(columns={"CLASS_LABEL": "Label"}, inplace=True)  # Rename label column to a consistent name

# STEP 2: Load Dataset 2 (Phising_Detection_Dataset.csv)
# -------------------------------------------------------
# This dataset has mostly URL-based features and is heavily imbalanced (more legit samples).
df2 = pd.read_csv("Phising_Detection_Dataset.csv")
df2 = df2.select_dtypes(include=["number"])            # Keep only numeric features
df2.rename(columns={"Phising": "Label"}, inplace=True) # Rename label column for consistency

# STEP 3: Combine the Two Datasets Using Shared Columns
# ------------------------------------------------------
# We take the intersection of column names so that both datasets align in structure.
common_columns = df1.columns.intersection(df2.columns)
df_combined = pd.concat([df1[common_columns], df2[common_columns]], axis=0)

# STEP 4: Clean the Data (Drop Rows with Missing Labels)
# -------------------------------------------------------
# If any row has a missing label (NaN), we drop it so training doesn't fail.
df_combined = df_combined.dropna(subset=["Label"])

# STEP 5: Balance the Dataset
# ----------------------------
# We balance the dataset by downsampling the majority class (usually legitimate).
class_counts = df_combined["Label"].value_counts()
df_class_0 = df_combined[df_combined["Label"] == 0]  # Phishing samples
df_class_1 = df_combined[df_combined["Label"] == 1]  # Legitimate samples

# If legitimate samples > phishing, randomly downsample legitimate to match phishing
if class_counts[1] > class_counts[0]:
    df_class_1 = resample(df_class_1, replace=False, n_samples=len(df_class_0), random_state=42)

# Combine both classes back together and shuffle the rows
df_balanced = pd.concat([df_class_0, df_class_1]).sample(frac=1, random_state=42)

# STEP 6: Split Data Into Features and Labels
# --------------------------------------------
X = df_balanced.drop("Label", axis=1)   # Input features
y = df_balanced["Label"]                # Output labels (0 = phishing, 1 = legitimate)

# STEP 7: Train/Test Split
# -------------------------
# We split the data into 80% training and 20% test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# STEP 8: Train a Random Forest Model
# ------------------------------------
# We use 200 trees to improve stability and accuracy.
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# STEP 9: Evaluate Model Accuracy and Print Metrics
# --------------------------------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# STEP 10: Save the Trained Model
# --------------------------------
# This model will be used in a Flask API or a browser extension backend.
joblib.dump(model, "rf_model_combined.pkl")

# STEP 11: Save the Feature Order
# --------------------------------
# This ensures that the feature vector used for prediction has the correct order.
with open("feature_order_combined.txt", "w") as f:
    for feature in X.columns:
        f.write(f"{feature}\n")

# STEP 12: Visualize Feature Importances (Top 15)
# -------------------------------------------------
# We use this to understand which features are most influential.
importances = model.feature_importances_            # Get feature importance scores
indices = importances.argsort()[::-1]               # Sort indices of features by importance descending
features = X.columns                                # List of feature names

# Create a bar plot for the top 15 features
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices][:15], y=features[indices][:15])
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance_plot.png")          # Save plot to file
plt.show()
