#questo e' un classificatore che usa un albero decisionale
#classificatore = box di regole

from sklearn import tree

#inputs del classificatore
#features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]
#0 = bumpy, 1 = smooth
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
#output che vogliamo
#labels = ["apple", "apple", "orange", "orange"]
#0 = apple, 1 = orange
labels = [0, 0, 1, 1]

#inizializzazione albero decisionale
clf = tree.DecisionTreeClassifier()
#l'algoritmo di training e' incluso nel classificatore (fit)
clf = clf.fit(features, labels)

#a questo punto abbiamo un classificatore gia trainato.
#primo test

print clf.predict([150, 0])

#il risultato sara 0 nel caso di riconoscimento di una mela, 1 per l'arancia
