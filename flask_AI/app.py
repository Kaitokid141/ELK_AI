from urllib import response
from flask import Flask, request, jsonify
import pickle
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

app = Flask(__name__)

class Detector:
    def __init__(self, graph):
        self.graph = graph
        with self.graph.as_default():
            self.tokenizer = pickle.load(open("./tokenizer/tokenizer.pickle", "rb"))
            self.model = keras.models.load_model("./model/cnn_clf.h5")
            #self.tokenizer = pickle.load(open("./tokenizer.pickle", "rb"))
            #self.model = keras.models.load_model("./cnn_clf.h5")
            self.max_len = 600
            self.labels_type = ['SQLi', 'anomalous', 'normal', 'XSS', 'SSI', 'BufferOverflow', 'CRLFi', 'XPath', 'LDAPi', 'FormatString']

    def props_to_labels(self, props_matrix):
        labels = []
        for props_vector in props_matrix:
            idx = np.argmax(props_vector)
            label = self.labels_type[idx]
            labels.append(label)
        return labels

    def predict_url(self, url):
        label_pred = self.predict_urls([url])[0]
        return label_pred

    def predict_urls(self, urls):
        seq = self.tokenizer.texts_to_sequences(urls)
        X = sequence.pad_sequences(seq, maxlen=self.max_len)
        with self.graph.as_default():
            Y_pred = self.model.predict(X)
        labels_pred = self.props_to_labels(Y_pred)
        return labels_pred

graph = tf.Graph()
detector = Detector(graph)   

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON'})
        else:
            input_text = data.get('text', '')
            #input_text = "/tienda1/publico/caracteristicas.jsp?id=d%27z%220"

            #detector = Detector()
            #response_text = detector.predict_url(input_text)
            with graph.as_default():
                response_text = detector.predict_url(input_text)
            return jsonify({'response': response_text})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000, debug=True)
