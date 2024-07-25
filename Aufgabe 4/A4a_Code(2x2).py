import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import time

st = time.time()

# Anzahl der Muster
total_patterns = 1000
checkerboard_patterns = 500

# Funktion, um ein zufälliges 2x2-Muster zu erstellen
def random_pattern():
    pattern = [0, 0, 1, 1]  # zwei schwarze (0) und zwei weiße (1) Felder
    np.random.shuffle(pattern)
    return pattern

# Funktion, um ein Schachbrettmuster zu erstellen
def checkerboard_pattern():
    return [0, 1, 1, 0]

# Listen für Muster und Labels
patterns = []
labels = []

# Erstelle die Schachbrettmuster
for _ in range(checkerboard_patterns):
    patterns.append(checkerboard_pattern())
    labels.append(1)

# Erstelle die zufälligen Muster
for _ in range(total_patterns - checkerboard_patterns):
    while True:
        pattern = random_pattern()
        # Überprüfen, ob es kein Schachbrettmuster ist
        if not (pattern == [0, 1, 1, 0] or pattern == [1, 0, 0, 1]):
            patterns.append(pattern)
            labels.append(0)
            break

# Zufälliges Mischen der Muster und Labels
combined = list(zip(patterns, labels))
np.random.shuffle(combined)
patterns[:], labels[:] = zip(*combined)

# Konvertieren der Listen in numpy Arrays
patterns_array = np.array(patterns)
labels_array = np.array(labels)

# Definieren des Modells
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))  # Erster Layer mit 8 Neuronen und ReLU
model.add(Dense(4, activation='relu'))  # Zweiter Layer mit 4 Neuronen und ReLU
model.add(Dense(1, activation='sigmoid'))  # Output Layer mit 1 Neuron und Sigmoid

# Kompilieren des Modells
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training des Modells
history = model.fit(patterns_array, labels_array, epochs=100, batch_size=10)

# Vorhersagen machen
predictions = model(patterns_array)

#
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.legend(['Accuracy', 'Loss'])
plt.show()