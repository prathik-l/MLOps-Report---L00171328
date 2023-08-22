Python 3.11.4 (tags/v3.11.4:d2340ef, Jun  7 2023, 05:45:37) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
... from sklearn.model_selection import train_test_split
... from sklearn.preprocessing import StandardScaler
... 
... # Load the Credit Card Fraud Detection dataset from Kaggle
... # You need to download the dataset from the Kaggle link and provide the appropriate file path
... data = pd.read_csv("creditcard.csv")
... 
... # Preprocessing
... X = data.drop("Class", axis=1)
... y = data["Class"]
... 
... # Split the data into training and testing sets
... X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
... 
... # Scale the features using StandardScaler
... scaler = StandardScaler()
... X_train_scaled = scaler.fit_transform(X_train)
... X_test_scaled = scaler.transform(X_test)
