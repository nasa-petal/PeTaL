from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Activation
from keras import backend as K
# from PIL import Image
import numpy as np
import tensorflow as tf
import petal.graph

class ResNet():
    def __init__(self, model='ResNet50'):
        self.model = ResNet50()
        self.graph = tf.get_default_graph()
    def classify(self,img, top=5):
        x = load_img(img, target_size=(224,224))
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        with self.graph.as_default():
            preds = self.model.predict(x)
        decoded_preds = decode_predictions(preds)
        top *= -1
        top_indices = preds[0].argsort()[top:][::-1]
        return [{'label':pred[1], 'score':'{:.3f}%'.format(pred[2]*100)} for pred in decoded_preds[0]]
    
class GeoNet():
    def __init__(self, model='ResNet50'):
        self.model = load_model('petal/models/geonet.h5', custom_objects={'relu6': Activation('relu')})
        self.graph = tf.get_default_graph()
        self.classes = ['branching', 'explosion', 'spiral', 'tile']
    def classify(self,img):
        x = load_img(img, target_size=(224,224), grayscale=True)
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        with self.graph.as_default():
            preds = self.model.predict(x)[0]
        scores = np.argsort(preds)
        return [{'label':self.classes[s], 'score':'{:.3f}%'.format(preds[s]*100)} for s in scores[::-1]]