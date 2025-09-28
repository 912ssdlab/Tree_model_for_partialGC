import pandas as pd
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,recall_score,f1_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

train_data = pd.read_csv('./train_data', index_col=0)
train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
test_data = pd.read_csv('./test_data', index_col=0)
print("shape is :", train_data.shape)
print(train_data.head())
print(train_data.info())

# pre handle
train_data["SB_update_count"] = train_data["SB_update_count"].apply(lambda x: 1 if x > 0 else 0)
test_data["SB_update_count"] = test_data["SB_update_count"].apply(lambda x: 1 if x > 0 else 0)
# split data
y_train = train_data["SB_update_count"].values
x_train = train_data.drop("SB_update_count", axis=1).values

y_test = test_data["SB_update_count"].values
x_test = test_data.drop("SB_update_count", axis=1).values

# Normalized
scaler = MinMaxScaler()
# transform
x_train_normalized = scaler.fit_transform(x_train)
x_test_normalized = scaler.transform(x_test)

# train
clf = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=5, max_features='sqrt', min_samples_split=2, min_samples_leaf=4)
clf.fit(x_train_normalized, y_train)

# dump(clf, 'model/model.joblib')
# exit()
# predict
y_pred = clf.predict(x_test_normalized)

# evaluate
train_score = clf.score(x_train_normalized, y_train)
test_score = clf.score(x_test_normalized, y_test)

print("train_score is {0:.3f}; test_score is {1:.3f}".format(train_score, test_score))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:\n", cm)

# recall and F1
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("recall:", recall)
print("F1:", f1)

# try more params
param_grid = {
    'max_depth': [3, 5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
}

dt = DecisionTreeClassifier(random_state=42)

# try different params
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# grid search
grid_search.fit(x_train_normalized, y_train)

# best params
print("best params:", grid_search.best_params_)
print("best score: {:.4f}".format(grid_search.best_score_))

# 获取最佳模型（可选）
best_dt = grid_search.best_estimator_

dump(clf, 'model/model.joblib')
exit()