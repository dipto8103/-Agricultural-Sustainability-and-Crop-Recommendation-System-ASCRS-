################################################################################
# Name: Shreyas Sawant
# Roll no.: I059
# Purpose: Predict Rainfall of current year by taking inputs of previous years
################################################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

def data_reader(file_path):
    crop_data = pd.read_csv(file_path)
    print(crop_data.head(5))
    return crop_data

def data_cleaner(crop_data):
    crop_data.drop(["temperature", "humidity"], axis=1, inplace=True)
    print(crop_data.head(5))

    # Encode the 'label' column
    label_encoder = LabelEncoder()
    crop_data['label'] = label_encoder.fit_transform(crop_data['label'])

    X = crop_data.drop(columns=['label'])
    y = crop_data['label']

    return X, y, label_encoder

def data_normalizer(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(X_train_scaled.shape, X_test_scaled.shape)
    
    return scaler, X_train_scaled, X_test_scaled

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    return model

def predict_rainfall(model, scaler, label_encoder):
    try:
        test_sample = np.array([[78, 43, 56, 4.5, 343.2]])
        test_sample_scaled = scaler.transform(test_sample)
        y_pred = model.predict(test_sample_scaled)

        # Inverse transform the predicted label to get the original categorical value
        predicted_crop = label_encoder.inverse_transform([int(round(y_pred[0]))])
        print("Predicted crop is:", predicted_crop)

    except Exception as e:
        print("Exception occurred:", e)

if __name__ == '__main__':
    file_path = "/Users/shreyassawant/mydrive/Shreyus_workspace/IITB_hackathon_code/crop_recommendation_konkan_maharashtra.csv"
    crop_data = data_reader(file_path)

    X, y, label_encoder = data_cleaner(crop_data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

    scaler, X_train_scaled, X_test_scaled = data_normalizer(X_train, X_test)

    model = train_model(X_train_scaled, y_train)

    # Save the model, scaler, and label_encoder for future use
    with open('rainfall_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
        model_file.close()

    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
        scaler_file.close()

    with open('label_encoder.pkl', 'wb') as label_encoder_file:
        pickle.dump(label_encoder, label_encoder_file)
        label_encoder_file.close()

    # Example of using the trained model to predict rainfall
    predict_rainfall(model, scaler, label_encoder)