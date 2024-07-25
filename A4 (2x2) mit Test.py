import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

def Data(total_patterns,chessboard_patterns): # Definiere eine Funktion, die einen Datensatz der gewünschten GRöße erstellt

    # Funktion, um ein zufälliges 2x2-Muster zu erstellen
    def random_pattern():
        pattern = [0, 0, 1, 1]  # zwei schwarze (0) und zwei weiße (1) Felder
        np.random.shuffle(pattern)
        return pattern

    # Funktion, um ein Schachbrettmuster zu erstellen
    def chessboard_pattern():
        return [0, 1, 1, 0]

    # Listen für Muster und Labels
    patterns = []
    labels = []

    # Erstelle die Schachbrettmuster
    for _ in range(chessboard_patterns):
        patterns.append(chessboard_pattern())
        labels.append(1)

    # Erstelle die zufälligen Muster
    for _ in range(total_patterns - chessboard_patterns):
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
    return patterns_array, labels_array

# Erzeuge Trainings- und Testdaten
Trainingdata = Data(1000, 500)
Testdata = Data(300, 80)

# Definieren des Modells
model = Sequential()
#model.add(Dense(8, input_dim=4, activation='relu'))  # Erster Layer mit 8 Neuronen und ReLU
model.add(Dense(4, activation='relu'))  # Zweiter Layer mit 4 Neuronen und ReLU
model.add(Dense(1, activation='sigmoid'))  # Output Layer mit 1 Neuron und Sigmoid

# Kompilieren des Modells
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training des Modells
history = model.fit(Trainingdata[0], Trainingdata[1], epochs=100, batch_size=10)

# Vorhersagen machen
predictions = model(Trainingdata[0])

#
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.legend(['Accuracy', 'Loss'])
plt.show()

# Testen, ob das Modell die Schachbretter gut erkennt
loss, accuracy = model.evaluate(Testdata[0], Testdata[1])
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
#print(model.summary())
#print(model.get_weights())