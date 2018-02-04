# importa il dataset su cui lavorare
from sklearn.datasets import load_iris #toy dataset from sklearn
from sklearn import tree
import numpy as np

iris = load_iris()

# lavori col dataset
#	print iris.feature_names 	printa le varie colonne
#	print iris.target_names		printa i possibili output
#	print iris.data[0]		printa la prima entry della tabella
#	print iris.target[0]		printa il primo nome associato alla
#					prima entry della tabella

# 	print "example %d: label %s, features %s" % (i, iris.target[i], iris.data[i])
# in questo modo pronto tutto il dataset

# esempio di rimozione dati
test_idx = [0, 50, 100]

# traina il classificatore
#	prima di tutto bisogna splittare i dati

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)
# testing data (esempi)
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print test_target

# predici la label per il nuovo fiore in input

print clf.predict(test_data)

# visualizza l'albero

# viz code
# 	import pydotplus
# 	dot_data = tree.export_graphviz(clf, out_file=None)
# 	graph = pydotplus.graph_from_dot_data(dot_data)
# 	graph.write_pdf("iris.pdf")

print test_data[0], test_target[0]
print iris.feature_names, iris.target_names


