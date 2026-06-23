import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

df = pd.read_csv("abalone.csv")
df['Sex'] = df['Sex'].map({'M': 0, 'F': 1, 'I': 2})

X = df.drop('Rings', axis=1)
y = df['Rings']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

dtrain = xgb.DMatrix(X_train, label=y_train)
params = {'objective': 'reg:squarederror', 'max_depth': 3}
model = xgb.train(params, dtrain, num_boost_round=100)
model.save_model("model.bst")
