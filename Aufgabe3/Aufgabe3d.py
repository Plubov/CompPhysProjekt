import Neurons as N
import numpy as np
import random

def TestChessboard(Chessboard,weights,korrekt):

    Neurons = N.setupNetwork(Chessboard) # Erstellung und Anregung der Input-Neuronen nach Brettvorlage

    V_Features_Out = [Neurons[0].get_V_max(), Neurons[1].get_V_max(), Neurons[2].get_V_max(),
                      Neurons[3].get_V_max()]  # Sämtliche äußere Spannungsmaxima der Input Neuronen

    I_5 = 0  # Letzendliche Äußere maximale Stromstärke an Output Neuron als Summe aller Spannungen der Feature Neuronen mult. mit ihren Gewichten

    for i in range(0, len(V_Features_Out)):  # Addiren aller Spannungen der Input Neuronen mult. mit ihrem Gewicht
        I_5 += weights[i] * V_Features_Out[i]

    if I_5 < -5:  # Verhinderung, dass I < I0 vorkommt
        I_5 = -5
    TargetNeuron = N.TargetNeuron(I_5)  #Initioalisierung Target-Neuron mit endgültiger angelegter Spannung

    print("Maximale Aktivierung des Target neurons: ",max(TargetNeuron.get_activation()))  # Ausgabe der Aktivierung (Erkennungswahrscheinlichkeit für Schachbrett) des Target-Neuurons

    if max(TargetNeuron.get_activation())>=.5:      #Rückgaben, ob das Schachbertt korrekterweise erkannt werden konnte
        if korrekt:
            return True
        else:
            return False
    else:
        if korrekt:
            return False
        else:
            return True

weights=np.zeros(4,dtype=float)     #Initioaliseren der Anfangsgewichte
for i in range(4):                          #Zufälliges Wählen der Anfangsgewichte mit Werten zw. 0 und 1
    weights[i] = random.randrange(0,1)

def training_round(Chessboard,init_weights,isChessboard):
    weights=init_weights

    TestChessboard(Chessboard,init_weights,isChessboard)
    if