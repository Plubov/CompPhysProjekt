import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import time

st = time.time() # Zeitnahme zum Startzeitpunkt
# Parameter
board_size = 8 # Größe des Bretts
total_patterns = 1000 # Anzahl der Muster, die erzeugt werden
checkerboard_patterns = 500 # Anzahl der Muster, die ein Schachbrett sind

# Funktion, um ein zufälliges Muster der Größe "board_size" zu erstellen
def random_pattern(board_size):
    pattern = np.zeros((board_size, board_size), dtype=int) # Initialisierung eines Arrays mit Größe: "Board_size"²
    white_positions = np.random.choice(board_size * board_size, board_size * board_size // 2, replace=False) # Markiere genau 32 Felder als weiße Felder
    pattern.flat[white_positions] = 1 # Setze den Index der Weißen Felder auf 1
    return pattern

# Funktion, um ein 8x8-Schachbrettmuster zu erstellen
def checkerboard_pattern(board_size):
    pattern = np.zeros((board_size, board_size), dtype=int) # Initialisierung eines Arrays mit Größe: "Board_size"²
    for i in range(board_size): # geht jede Zeile durch
        for j in range(board_size): # geht jede Spalte durch und somit zusammen ds gesamte Feld
            pattern[i, j] = (i + j) % 2 # Setzt den Index im Array entsprechend eines Schachbrettmusters
    return pattern

# Erzeugen von Listen für Muster und Labels
patterns = []
labels = []

# Erstelle je eine Liste mit Aussage, dass es sich um Schabrettmuster handelt und weitere Liste mit Schabrettmuster
for _ in range(checkerboard_patterns):
    patterns.append(checkerboard_pattern(board_size)) # Füge alle Schachbrettmuster in die "Patterns" Liste
    labels.append(1) # Füge alle Wahrheitaussagen (Schachbrett = 1) in die "labels" Liste

# Erstelle die zufälligen Muster
for _ in range(total_patterns - checkerboard_patterns): # Lege die Anzahl der zufälligen Muster fest
    while True:
        pattern = random_pattern(board_size) # Sicherstellen, dass keine Schachbrettmuster hinzugefügt werden
        if not np.array_equal(pattern, checkerboard_pattern(board_size)):
            patterns.append(pattern) # Füge alle Muster in bestehende Liste mit Mustern hinzu
            labels.append(0) # Füge alle Wahrheitsaussagen (kein Schachbrettmuster = 0) zur bestehenden Liste hinzu
            break

# Zufälliges Mischen der Muster und Labels
combined = list(zip(patterns, labels))
np.random.shuffle(combined)
patterns[:], labels[:] = zip(*combined)

# Konvertieren der Listen in numpy Arrays und Flatten der Muster
patterns_array = np.array(patterns).reshape(total_patterns, -1)
labels_array = np.array(labels)

# Definieren des Modells
model = Sequential()
model.add(Dense(64, input_dim=board_size * board_size, activation='relu'))  # Erster Layer mit 64 Neuronen und ReLU
model.add(Dense(32, activation='relu'))  # Zweiter Layer mit 32 Neuronen und ReLU
model.add(Dense(1, activation='sigmoid'))  # Output Layer mit 1 Neuron und Sigmoid

# Kompilieren des Modells
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training des Modells
history = model.fit(patterns_array, labels_array, epochs=5, batch_size=10)

# Vorhersagen machen
predictions = (model.predict(patterns_array) > 0.5).astype("int32")

# Ausgabe von Beispielen
for i in range(10):
    print(f"Pattern {i+1}:\n", patterns_array[i].reshape(board_size, board_size))
    print(f"Prediction: {predictions[i][0]}, Actual: {labels_array[i]}\n")

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.legend(['Accuracy', 'Loss'])
plt.show()
print('Laufzeit:', time.time()-st )