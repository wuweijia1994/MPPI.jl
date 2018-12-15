# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.5
# ---

# +
import os
# import tensorflow as tf
# from tensorflow import keras
from keras.models import model_from_json, Sequential
from keras.layers import Dense
import keras

def create_optimizer():
    return keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)        
    
def create_model_from_scratch(state_dimention, action_dimention):
    model = Sequential()
    
    layer = Dense(units=32, activation='relu', input_dim=state_dimention, kernel_initializer='zeros', bias_initializer='zeros')
    model.add(layer)
#     print("first layer: {}".format(layer.get_weights()))
    
    layer = Dense(units=32, activation='relu', kernel_initializer='zeros', bias_initializer='zeros')
    model.add(layer)
#     print("first layer: {}".format(layer.get_weights()))
    
    layer = Dense(units=action_dimention, kernel_initializer='zeros', bias_initializer='zeros')
    model.add(layer)
#     print("first layer: {}".format(layer.get_weights()))

    optimizer = create_optimizer()
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    
    return model

def load_model(model_name):
    # load json and create model
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name+".h5")

    optimizer = create_optimizer()
    loaded_model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    
    return loaded_model
    print("Loaded model from disk")
    
def save_model(model, model_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name+".h5")
    print("Saved model to disk")

def evaluate_model(model, X, Y, verbose = 0):
    print(X)
    print(Y)
    scores = model.evaluate(X, Y, verbose)
    return scores

def build_model(RESUME, model_name, state_dimention = 64, action_dimention=1):
    if RESUME:
        if os.path.isfile(model_name+".json"):
            return load_model(model_name)
        else:
            return create_model_from_scratch(state_dimention, action_dimention)
            print("No saved model called " + model_name + " found!")
            print("Create memory from scratch.")
    else:    
        return create_model_from_scratch(state_dimention, action_dimention)
        
        
if __name__  == "__main__":
    model = build_model()
    model.summary()
