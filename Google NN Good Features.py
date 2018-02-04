import numpy as np
import matplotlib.pyplot as plt

# popolazione
greyhounds = 500
labs = 500

# aggiungo random 4 inches di altezza ai vari cani
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

# istogramma.
plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'] )
plt.show()

# dall'istogramma si nota che tra le varie altezze quelle di mezzo restano dubbie
# non i puo assumere con certezza che si tratti di un grayhound o un labrador
# questo e' il motivo per il quale servono più features che sappiano
# splittare i dati in una maniera migliore.
# Ad esempio usando la lungheza del pelo, la velocita media, il peso medio ecc in
# aggiunta all'altezza.

# features indipendenti
# evitare ridondanze
# facili da capire



