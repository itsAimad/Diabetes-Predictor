import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


def create_model(data):
        # Define features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Normalize the data
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression(random_state=42,n_jobs=2)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    return model,scaler


def clean_data():
    df = pd.read_csv("Data/diabetes.csv")
    columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Replace zero values with the mean of the respective column
    for column in columns_with_zero:
        df[column].replace(0, df[column].mean(), inplace=True)

    return df




def main():
    data = clean_data()
    model,scaler = create_model(data)

    with open("model/model.pkl","wb") as f:
        pickle.dump(model,f)

    with open("model/scaler.pkl","wb") as f:
        pickle.dump(scaler,f)


if __name__ == "__main__":
    main()