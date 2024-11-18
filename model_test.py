import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Data Import
data = load_diabetes()

# Define X and y
X = pd.DataFrame(data['data'],
                  columns=data['feature_names'])

y = pd.Series(data['target'])

# Train test split
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.33, random_state=42)


# Pipeline definition

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Preprocessing step: standardize the features
    ('regressor', LinearRegression())  # Model step: linear regression
])

# Train the pipeline on the training data
pipeline.fit(X, y)




