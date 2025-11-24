# importing libraries
import csv
import os
import pickle
import random
from io import StringIO
import boto3
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

# defining the environment variables
data_s3_url = os.environ['data_s3_url']
model_bucket_name = os.environ['model_bucket_name']


# defining the function that imports the dataset
def read_csv_from_s3(s3_url):
    bucket_name = s3_url.split('/')[2].split('.')[0]
    file_key = '/'.join(s3_url.split('/')[3:])
    s3 = boto3.client('s3')
    csv_obj = s3.get_object(bucket=bucket_name, key=file_key)
    csv_string = csv_obj['body'].read().decode('utf-8')
    csv_reader = csv.reader(StringIO(csv_string), delimiter=':')
    headers = next(csv_reader)
    data = [row for row in csv_reader]
    return headers, data


# defining a function that splits the dataset into the training set and the test set
def split_data(data, test_ratio=0.10):
    np.random.shuffle(data)
    split_index = int(len(data) * (1 - test_ratio))
    return data[:split_index], data[split_index:]


# Defining a function that prepares the data for feature extraction and target variable separation
def prepare_data(headers, data):
    x = [list(map(float, row[:-1])) for row in data]
    y = [float(row[-1]) for row in data]
    return x, y


# defining a function that trains a random Forest model on the training  set, evaluates it on the test set and saves
# it on s3 bucket
def train_evaluate_save(event, context):
    headers, raw_data = read_csv_from_s3(data_s3_url)
    train_data, test_data = split_data(raw_data)
    X_train, y_train = prepare_data(headers, train_data)
    X_test, y_test = prepare_data(headers, test_data)
    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(X_train, y_train)
    model_name = 'Random-Forest-Model-' + str(random.randint(0, 100))
    temp_file_path = '/tmp/' + model_name
    with open(temp_file_path, 'wb') as f1:
        pickle.dump(regressor, f1)
    with open(temp_file_path, 'rb') as f2:
        model_data = f2.read()
    s3 = boto3.resource('s3')
    s3_object = s3.Object(model_bucket_name, model_name)
    s3_object.put(Body=model_name)
    y_predicted = regressor.predict(X_test)
    print("Mean absolute error: ".format(metrics.mean_absolute_error(y_test, y_predicted)))
    return model_name


# Defining a function that Predicts the quality of a chosen wine
# using pretrained model saved in s3
def predict_with_quality(event, context):
    input_date = [
        float(event['fixed acidity']),
        float(event['volatile acidity']),
        float(event['citric acid']),
        float(event['residual sugar']),
        float(event['chlorides']),
        float(event['free sulfur dioxide']),
        float(event['total sulfur dioxide']),
        float(event['density']),
        float(event['pH']),
        float(event['sulphates']),
        float(event['alcohol'])
    ]
    model_name = event['model name']
    temp_file_path = '/tmp/'+model_name
    s3 = boto3.client('s3')
    s3.download_file(model_bucket_name, model_name, temp_file_path)
    with open(temp_file_path, 'rb') as f:
        model = pickle.load(f)
    return str(round(model.predict([input_date])[0], 1))




