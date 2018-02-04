# per lavorare con le immagini il deep learning da risultati migliori
# si lavore con i pixel grezzi delle immagini senza estrarre features a mano

from sklearn import metrics, cross_validation
import tensorflow as tf
from tensorflow.contrib import learning


def main(unused_argv):
    # carico il dataset
    iris = learn.datasets.load_dataset('iris')
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)

    # costruisco 3 livelli di Deep NN con 10, 20, 10 unita
    classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)

    # funzioni di fit e predict
    classifier.fit(x_train, y_train, steps=200) # train

    score = metrics.accuracy_score(y_test, classifier.predict(x_test))
    										# classifico i nuovi dati
    print('Accuracy: {0:f}', format(score))
    
