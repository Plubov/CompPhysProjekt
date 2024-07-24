import Neurons as N
import numpy as np
import random

def TestChessboard(Chessboard,weights,korrekt):         #Methode zur Überprüfung ob ein Schachbrett korrekt identifiziert/nicht identifiziert wird (1DArray,1DArray,Boolean)

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

def TestAllChessboards(Chessboards,weights,areChessboards):    #Funktion zur Überprüfung , ob das gesamte Set fehlerfrei erkannt werden konnte (2DArray, 1DArray, 1DArray)
    fehlerfrei=True
    for i in range(len(Chessboards)):       #Durchgehen aller Chessboard um zu überprüfen, ob einen falsch erkannt wurde
        if TestChessboard(Chessboards[i],weights,areChessboards[i]):    #Fall, dass Board korrekt erkannt wurde
            fehlerfrei=True
        else:                                                           #Fall, dass Board Fehlidentifiziert wurde
            fehlerfrei=False
    return fehlerfrei                                                   #Rückgabe, ob alle Boards korrekt erkannt werden konnten


weights=np.zeros(4,dtype=float)     #Initioaliseren der Anfangsgewichte
for i in range(4):                          #Zufälliges Wählen der Anfangsgewichte mit Werten zw. 0 und 1
    weights[i] = random.randrange(0,1)

#Trainingschessboards
chessboards=[
[0,0,1,1],      #False
[0,1,1,0],      #True
[0,1,0,1],      #False
[0,1,1,0],      #True
[1,0,0,1],      #False
[0,1,1,0],      #True
[1,0,1,0],      #False
[0,1,1,0],      #True
[1,1,0,0],      #False
[0,1,1,0],      #True
]
isChessboard=[False,True,False,True,False,True,False,True,False,True]#Liste ob Element aus chessboards echtes Schachbrettmuster ist


def training(Chessboards,init_weights,areChessboard):       #Trainingsmethode, welche solange Gewichte anpasst, bis alle Boards fehlerfrei erkannt werden und die finalen
    currentWeights=init_weights
    runs =0
    while (not TestAllChessboards(Chessboards,currentWeights,areChessboard)):
        currentWeights=(currentWeights,Chessboards,areChessboard)
        runs+=1
    print("Finale Gewichte: ",currentWeights)
    print("Total runs needed:",runs)
    

def adjustWeights(currentWeights,Chessboards,areChessboard):








training(chessboards,weights,isChessboard)      #Aufrufen der Trainigsmethode