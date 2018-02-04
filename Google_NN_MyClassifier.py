# Scrivere da 0 un classificatore
from scipy.spatial import distance

class ScrappyKNN():

	# metodi necessari nel classif.

	def fit(self, X_train, y_train):
		self.X_train = X_train # store delle variabili
		self.y_train = y_train

	def predict(self, X_test):
		# X_test è un array 2d
 		predictions = []
 		# ogni riga contiene le features di un testing example
 		for row in X_test:
       		label = self.closest(row)
         	predictions.append(label)
 		return predictions

 		# distanza euclidea tra due punti
	def euc(a, b):
		return distance.euclidean(a, b)

 	def closest(self, row):
      	# calcolo la distanza dal punto di test al primo training point
      	# tengo traccia della distanza minore trovata fino ad ora
  		best_dist = euc(row, self.X_train[0])
    	# indice del training point più vicino
    	best_index = 0
    	# itero su tutti gli altri training point
    	# ogni volta che ne trovo uno piu vicino, updato
    	for i in range(1, len(self.X_train)):
        	dist = euc(row, self.X_train[i])
         	if dist < best_dist:
            	best_dist = dist
            	best_index = i
        # uso l'index per ritornare la label del training point piu vicino
        return self.y_train[best_index]
