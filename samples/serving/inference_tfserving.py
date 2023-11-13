import numpy as np
from PIL import Image
import requests

import numpy as np
from PIL import Image
    
    
def jpeg_to_mnist_array(image_path):
    try:
    # Open the image using PIL
        with Image.open(image_path) as image:
            # Resize the image to 28x28 pixels
            image_resized = image.resize((28, 28))

            # Convert the image to grayscale (L mode)
            image_gray = image_resized.convert("L")

            # Normalize pixel values to the range [0, 1]
            image_normalized = np.array(image_gray) / 255.0

            # Reshape the image to the MNIST format (28x28x1)
            mnist_array = image_normalized.reshape(28, 28, 1)
            
    except Exception as e:
            raise ValueError("Failed to process the image:", str(e))

    return mnist_array
    

#setting endpoint
name="digits-recognizer"
namespace = utils.get_default_target_namespace()

url="http://{}.{}.svc.cluster.local/v1/models/{}:predict".format(name,namespace,name)

#load image e formatting request

#loading image
image_path="./4.png"
array_test = jpeg_to_mnist_array(image_path)
array_test = array_test.reshape(-1,28,28,1)

#formatting json
data_formatted = np.array2string(array_test, separator=",", formatter={"float": lambda x: "%.1f" % x})
json_request = '{{ "instances" : {} }}'.format(data_formatted)

#sending request
response = requests.post(url, data=json_request)

print("Prediction for the image")
json_response = response.json()
print(json_response)

print("Predicted: {}".format(np.argmax(json_response["predictions"])))
