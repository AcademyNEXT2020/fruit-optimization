import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np


IMG_HEIGHT = 100
IMG_WIDTH = 100

def preprocess():
    # Generator for given test data, normalizes the data
    test_image_generator = ImageDataGenerator(rescale=1./255)
    # Generator for hidden test data, normalizes the data
    hidden_test_image_generator = ImageDataGenerator(rescale=1./255)

    # assign variables with the proper file path for given and hidden test sets
    test_dir = os.path.join(os.getcwd(), 'test')
    hidden_test_dir = os.path.join(os.getcwd(), 'test2')

    # convert all the images in a directory into a format that tensorflow
    # can work with
    test_data_gen = test_image_generator.flow_from_directory(
                    directory=test_dir,
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    class_mode='categorical')
    hidden_test_data_gen = hidden_test_image_generator.flow_from_directory(
                    directory=hidden_test_dir,
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    class_mode='categorical')

    return (test_data_gen, hidden_test_data_gen)


def score_model(test_data_gen, hidden_test_data_gen, filename):
    # load saved model
    model = tf.keras.models.load_model(filename)
    # evaulate the model using the given test set
    test_loss, test_accuracy = model.evaluate(test_data_gen)
    # evaulate the model using the hidden test set
    hidden_test_loss, hidden_test_accuracy = model.evaluate(hidden_test_data_gen)
    print("Accuracy", test_accuracy, hidden_test_accuracy)


(test_data_gen, hidden_test_data_gen) = preprocess()
# replace this string with your saved model .h5 filename
filename = ''
score_model(test_data_gen, hidden_test_data_gen, filename)
