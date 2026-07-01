import os
import pandas as pd
import numpy as np 
import kagglehub
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


DARKEST = "#344E41"
DARK = "#3A5A40"
MID = "#588157"
LIGHT = "#5A9F68"
LIGHTEST = "#BBD58E"

# Load the Dataset
path = kagglehub.dataset_download("aldinwhyudii/student-depression-and-lifestyle-100k-data")
csv_file = os.path.join(path, 'student_lifestyle_100k.csv')
df = pd.read_csv(csv_file)


# Understand the data (Check)
df.info()
print(df.head())
print(df.describe())
print(df.isnull().sum())
print(df.isna().sum())
print(df.duplicated().sum())
# (No duplicated and no null or missing value + the outlier is meaningful this is why i didnot remove it)


# Drop and one-hot-encoding
df = df.drop(columns=['Student_ID'])
df = pd.get_dummies(df, columns=['Department'], dtype=int)


# Feature eng ( The best part for me )
Sleep_time = 8
df['Sleep_effect'] = (Sleep_time - df['Sleep_Duration']).clip(lower=0.5)
df['Sleep_Stress_relation'] = df['Sleep_effect'] * df['Stress_Level']
df['Study_Sleep_ratio'] = df['Study_Hours'] / df['Sleep_Duration']
df['Gender'] = df['Gender'].map({'Female': 0,'Male': 1 })
df['Depression'] = df['Depression'].astype(int)


# Split the data
X = df.drop(columns=['Depression'])
y = df['Depression']


# train test split with stratify becouse the Data is imbalanced
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Scale The Data
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

# PCA 
pca = PCA(n_components=0.95)
X_trainpca = pca.fit_transform(X_train)
X_testpca = pca.transform(X_test)
ratio = pca.explained_variance_ratio_
print (ratio[:5].round(3))


# RandomForest
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

#Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_pca = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_pca.fit(X_trainpca, y_train)
lr_pred_pca = lr_pca.predict(X_testpca)

rf_para = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [1, 5, 10, 15, 25, 40],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf' : [1, 2, 4]
}

# # it freezing my laptop 
# rf_search = RandomizedSearchCV(
#         RandomForestClassifier(random_state=42, class_weight='balanced'),
#         param_distributions= rf_para,
#         n_iter=25,
#         cv=5, 
#         scoring='recall',
#         random_state=42, 
#         n_jobs=-1
# )

# rf_search.fit(X_train, y_train)
# rf_tune = rf_search.best_estimator_
# rf_tune_pred = rf_tune.predict(X_test)

# result 
# Logistic Regression
print('\nLogistic Regression')
print('Accuracy :', accuracy_score(y_test, lr_pred))
print('Precision:', precision_score(y_test, lr_pred))
print('Recall   :', recall_score(y_test, lr_pred))
print("F1 Score :", f1_score(y_test, lr_pred))


# random Fourest
print('\nRandom Forest')
print('Accuracy :', accuracy_score(y_test, rf_pred))
print('Precision:', precision_score(y_test, rf_pred))
print('Recall   :', recall_score(y_test, rf_pred))
print("F1 Score :", f1_score(y_test, rf_pred))

# pca
print('\nLogistic Regression with PCA')
print('Accuracy :', accuracy_score(y_test, lr_pred_pca))
print('Precision:', precision_score(y_test, lr_pred_pca))
print('Recall   :', recall_score(y_test, lr_pred_pca))
print("F1 Score :", f1_score(y_test, lr_pred_pca))

# # random Fourest Wtih random search
# print('\nRandom Forest (tuned, RandomizedSearchCV)')
# print('Accuracy :', accuracy_score(y_test, rf_tune_pred))
# print('Precision:', precision_score(y_test, rf_tune_pred))
# print('Recall   :', recall_score(y_test, rf_tune_pred))


# Most students sleep around 7 hours
plt.figure(figsize=(6, 4))
sns.histplot(df["Sleep_Duration"], kde=True, color=MID, edgecolor=DARKEST)
plt.title("Distribution of Sleep Duration")
plt.xlabel("Sleep Duration")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Most students between 2.3 and 3.7
plt.figure(figsize=(6, 4))
sns.histplot(df["CGPA"], kde=True, color=LIGHT, edgecolor=DARKEST)
plt.title("Distribution of CGPA")
plt.xlabel("CGPA")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Most students have stress between 3 and 5
plt.figure(figsize=(6, 4))
sns.histplot(df["Stress_Level"], kde=True, color=DARK, edgecolor=DARKEST)
plt.title("Distribution of Stress Level")
plt.xlabel("Stress Level")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Relationships between all numerical features
df_heatmap = df.copy()
df_heatmap["Depression"] = df_heatmap["Depression"].astype(int)
numeric_df = df_heatmap.select_dtypes(include=["number"])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(22, 5))
label = ["No Depression", "Depression"]
model_pred = [    
    ("Logistic Regression", lr_pred),
    ("Random Forest (baseline)", rf_pred),
    ("Logistic Regression (PCA)", lr_pred_pca)
    # ("Random Forest (tuned)", rf_tune_pred)
    ]
for ax, (title, pred) in zip(axes, model_pred):
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=label).plot(ax=ax, colorbar=False)
plt.tight_layout()
plt.show()


# Conclusion: (Before RandomSearchCV becouse my laptop cant handel it take while)
# Logistic Regression and Logistic Regression with PCA performed best for detecting depression, Randomforest have the high accuracy but low recall so can miss a real deprestion
# PCA did not significantly improve performance because the metrics before and after PCA were almost identical. 
# Since around 11–12 components were needed to explain 95% of the variance from 13 original features, PCA provided little dimensionality reduction.
# If i had more time and better laptop i think i will train RandomForest with RandomSearchCV and see how it goes