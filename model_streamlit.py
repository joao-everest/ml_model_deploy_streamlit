import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

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


# Step 2: Build Streamlit interface to get user input for prediction
st.title('Diabetes Prediction')
st.write("Enter the values for the features below:")

# Create input fields for each feature
# (For simplicity, we’ll use some of the Boston dataset's features. You can replace them with yours)
feature_names = X.columns

user_inputs = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0, step=0.01)
    user_inputs.append(value)

# Step 3: Make prediction when the user submits the input
if st.button('Predict Price'):
    # Convert the user inputs into a DataFrame
    input_data = np.array(user_inputs).reshape(1, -1)
    
    # Make a prediction using the trained pipeline
    prediction = pipeline.predict(input_data)
    
    # Display the predicted house price
    st.write(f"Predicted House Price: ${prediction[0]:,.2f}")
