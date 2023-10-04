
import numpy as np
from tensorflow import keras

def model_building(no_epochs:int , optimizer: str):
    """
    Build the model with Keras API
    Export model parameters
    """
    #dataset loading
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # reshaping the data
    # reshaping pixels in a 28x28px image with greyscale, canal = 1. This is needed for the Keras API
    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)
    # normalizing the data
    # each pixel has a value between 0-255. Here we divide by 255, to get values from 0-1
    x_train = x_train / 255
    x_test = x_test / 255
    
    #model definition
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(keras.layers.MaxPool2D(2, 2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))

    model.add(keras.layers.Dense(10, activation='softmax'))
    
    summary = model.summary()
    
    #compile the model - we want to have a binary outcome
    model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
    
    
    #fit the model and return the history while training
    history = model.fit(
      x=x_train,
      y=y_train,
      epochs=no_epochs,
      batch_size=20,
    )
    
    # Test the model against the test dataset
    # Returns the loss value & metrics values for the model in test mode.
    model_loss, model_accuracy = model.evaluate(x=x_test,y=y_test)
    
    
    # Generates output predictions for the input samples.
    test_predictions = model.predict(x=x_test)
    
    # Returns the indices of the maximum values along an axis.
    test_predictions = np.argmax(test_predictions,axis=1) 
    # the prediction outputs 10 values, we take the index number of the highest value, which is the prediction of the model


    #save trained model to minio
    keras.models.save_model(model,"tmp/detect-digits")

    #output the metrics to stdout
    loss_str = "loss="+str(model_loss)
    acc_str = "accuracy="+str(model_accuracy)
    print(loss_str)
    print(acc_str)
    

def main():
    #reading the parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_epochs', type=int)
    parser.add_argument('--optimizer', type=str)
    args = parser.parse_args()
    print("arguments:",args)

    model_building(args.no_epochs,args.optimizer)

if __name__ == "__main__":
    main()