import numpy as np
from kserve import utils
import requests
import pandas as pd
import json

def send_sklearn_inference_request(model_endpoint, x_array):
    '''
    send a request to the sklearn inference service using v2 format
    '''

    # Prepare the request payload in the required format
    x_list = x_array.tolist()
    data_formatted = np.array2string(x_array, separator=",", formatter={"float": lambda x: "%.1f" % x})
    payload = {
  
    "inputs" : [
        {
        "name" : "input0",
        "shape" : x_array.shape,
        "datatype" : "UINT32",
        "data" : x_list
        }
    ],
    "outputs" : [
        {
        "name" : "output0"
        }
    ]
}

    # Set the headers for the request
    headers = {"Content-Type": "application/json"}

    try:
        # Send the POST request to the scikit-learn inference service
        response = requests.post(model_endpoint, json=payload, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse and return the JSON response
            return response.json()
        else:

            print(f"Error: {response.status_code}")
            json_response = response.json()
            print(json_response)
            return None
    except Exception as e:
        print(f"Error sending request: {str(e)}")
        return None
    
#load dataset
df= pd.read_csv("bonfiglioli_pulito.csv")

#dataset preprocessing
df=df.drop('SERIALE', axis=1)
colonne_con_zeri = df.columns[(df == 0).all()]
df = df.drop(colonne_con_zeri, axis=1)
y_df = df['ESITO_AND_FINALE']
df = df.drop('ESITO_AND_FINALE', axis=1)

#using first row as test
print("Y true=",y_df[0])
x_array = df[0:1].to_numpy()

#setting endpoint
name = "model"
url = "http://10.152.183.201/v2/models/{}/infer".format(name)

if send_sklearn_inference_request(url,x_array)!= None:
    print("Y predicted=",send_sklearn_inference_request(url,x_array)['outputs'][0]['data'])
else:
    print("Error")