import numpy as np
import random

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
        # Überprüfen, ob es kein Schachbrettmuster ist (keine zwei benachbarten Felder dürfen gleich sein)
        if not (pattern == [0, 1, 1, 0] or pattern == [1, 0, 0, 1]):
            patterns.append(pattern)
            labels.append(0)
            break

# Zufälliges Mischen der Muster und Labels
combined = list(zip(patterns, labels))
random.shuffle(combined)
patterns[:], labels[:] = zip(*combined)

# Konvertieren der Listen in numpy Arrays, geeignet für TensorFlow
patterns_array = np.array(patterns).reshape(total_patterns, 2, 2)
labels_array = np.array(labels)




