# esempi sulle pipeline tra classificatori

# import di un dataset
from sklearn import datasets
import Google_NN_myClassifier
iris = datasets.load_iris()

X = iris.data	# features
y = iris.target	# label

# partizioniamo X e y in 2 set:
# X_train e y_train sono appunto per il train
# X_test e y_test sono per il testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)
# test_size = .5 perche voglio solo la meta dei dati per il test e
# l'altra meta per il train

# test del mio classificatore
my_classifier = ScrappyKNN()

# nuovo metodo di testing con i clasificatori
# from sklearn.neighbors import KNeighborsClassifier
# my_classifier = KNeighborsClassifier() #funziona esattamente allo stesso modo

# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier() # creo un classificatore

# se si vuole importare un qualsiasi altro classificatore, basta cambiare quelle due righe
# il codice funzionera esattamente allo stesso modo


my_classifier.fit(X_train, y_train) # traino il classificatore

predictions = my_classifier.predict(X_test) # provo una predizione sulle label

# print predictions
# [1 0 2 0 0 2 1 1 1 0 1 0 2 0 1 0 0 1 2 0 1 0 1 0 0 1 0 2 1 2 1 1 0 2 0 2 1
# 2 2 1 2 0 2 2 1 2 1 2 2 0 1 2 0 0 2 0 2 0 2 0 1 1 2 0 1 2 2 2 0 2 0 1 2 1
# 0]

# confrontiamo l'accuratezza del nostro classificatore comparando la lables predetta
# con la "true lable" (y_test)

from sklearn.metrics import accuracy_score
# print accuracy_score(y_test, predictions)
# 0.946666666667 (treeClassifier)
# 0.973333333333 (KNeighbors)


