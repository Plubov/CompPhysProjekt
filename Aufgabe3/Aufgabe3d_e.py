import Neurons as N
import numpy as np
import matplotlib.pyplot as plt
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

    #print("Maximale Aktivierung des Target neurons: ",max(TargetNeuron.get_activation()))  # Ausgabe der Aktivierung (Erkennungswahrscheinlichkeit für Schachbrett) des Target-Neuurons

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
    success=True
    for i in range(len(Chessboards)):       #Durchgehen aller Chessboard um zu überprüfen, ob einen falsch erkannt wurde
        if TestChessboard(Chessboards[i],weights,areChessboards[i]):    #Fall, dass Board korrekt erkannt wurde
            print("Success! Schachbrett ",i," korrekt identifiziert")
        else:                                                           #Fall, dass Board Fehlidentifiziert wurde
            print("Fail! Schachbrett",i,"falsch Identifiziert")
            success = False
        if success:
            print("Test erfolgreich!")  #Rückgabe, ob alle Boards korrekt erkannt werden konnten
        else:
            print("Test nicht erfolgreich!")

def quick_check(Chessboards,weights,areChessboards): #Schnellere Methode wie TestAllChessboards, nur ohne einzenle Zwischenergebnisse
    success = True
    for i in range(len(Chessboards)):  # Durchgehen aller Chessboard um zu überprüfen, ob einen falsch erkannt wurde
        if not TestChessboard(Chessboards[i], weights, areChessboards[i]):  # Fall, ein Board fehlerhaft erkannt wurde
            success = False
    if success:
        print("Test erfolgreich!")  # Rückgabe, ob alle Boards korrekt erkannt werden konnten
    else:
        print("Test nicht erfolgreich!")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def train_nn(initial_weights, training_boards, targets, learning_rate=0.1, epochs=10000):
    # Initialisiere die Gewichte
    weights = np.array(initial_weights)

    for epoch in range(epochs):
        for board, target in zip(training_boards, targets):
            # Forward Pass
            z = np.dot(board, weights)
            prediction = sigmoid(z)

            # Fehler berechnen
            error = target - prediction

            # Rückwärtspropagation
            adjustments = error * sigmoid_derivative(z)

            # Gewichte anpassen
            weights += learning_rate * adjustments * np.array(board)

    return weights
def train_nn_history(initial_weights, training_boards, targets, learning_rate=0.1, epochs=10000):#Trainingsmethode, welche das Plotten der gemachten fehler für Aufgabe 3e) ermöglicht
    # Initialisiere die Gewichte

    #Modifizierung zur Fehlerplottung
    error=0
    weights=np.copy(initial_weights)
    x_achse=np.linspace(1,epochs,epochs)
    weight_history=[]
    errors=[]
    for epoch in range(epochs):
        for board, target in zip(training_boards, targets):
            # Forward Pass
            z = np.dot(board, weights)
            prediction = sigmoid(z)

            # Fehler berechnen
            error = target - prediction

            # Rückwärtspropagation
            adjustments = error * sigmoid_derivative(z)

            # Gewichte anpassen
            weights += learning_rate * adjustments * np.array(board)
        weight_history.append(np.copy(weights))
        errors.append(error)


    return weights,x_achse,errors,weight_history
def gen_False_board():     #Funktion zur zufälligen erzeugung eines Schachbretts mit rückgabe, ob es ei korrektes Schachbrett ist
    pattern = [0, 1, 1, 0]  # Wahres Feld

    np.random.shuffle(pattern)  #Verändert das Schachbrett, sodass es kein wahres Feld mehr ist aber noch 2 1eun und nullen besitzt
    return pattern



def make_training_set(size=100,true_boards=50):     #Funktion, welche ein Trainingsset (Chessboards und Wahrheitsliste) mit gewünschter Größe und Anteil wahrer Boards enthält
    chessboards=np.zeros((size,4))      #Initialisierung leerer Chessboards
    arechessboards=[]               #Initialisierung leerer Wahrheitsliste
    for i in range(size):
        arechessboards.append(False)
    already_set=[]                      #Liste zur überprüfung ob Element aus chessboads bereits generiert wurde
    for i in range(0,true_boards):      #Initiales Auffüllen aller wahren boards an zufälligen stellen
        spot=random.randint(0,size-1)#Spot zum Auffüllen in chessboards
        while spot in already_set:      #Überpfrüfung, ob spot bereits platziert wurde, wenn ja dann suche neuen spot
            spot=random.randint(0,size-1)
        chessboards[spot]=[0,1,1,0]  #Platzierung von wahrem Board in spot
        already_set.append(spot)                        #Markierung, dass chessboard an spot bereits aufgefüllt wurde
    still_left=[]                       #Orte die noch aufgefüllt werden müssen
    for i in range(size):
        if i not in already_set:         #Wenn in i noch nicht gesetzt wurde soll dies in liste noch zu setzender Felder hinzugefügt werden
            still_left.append(i)

    for i in still_left:      #Restliches Auffüllen aller falschen Boards
        spot=random.randint(0,size-1)#Spot zum Auffüllen in chessboards
        while spot in already_set:      #Überpfrüfung, ob spot bereits platziert wurde, wenn ja dann suche neuen spot
            spot=random.randint(0,size-1)
        chessboards[spot]=gen_False_board()  #Platzierung von falschem Board
        already_set.append(spot)        #Markierung, dass chessboard an spot bereits aufgefüllt wurde
    for i in range(size):           #Erstellen der Wahrheitsliste ob Board auch Schachbrettmuster ist
        if np.array_equal(chessboards[i],[0,1,1,0]):
            arechessboards.append(True)
        else:
            arechessboards.append(False)

    return chessboards,arechessboards





weights=np.zeros(4,dtype=float)     #Initialiseren der Anfangsgewichte
for i in range(4):                          #Zufälliges Wählen der Anfangsgewichte mit Werten zw. 0 und 1
    weights[i] = random.uniform(0.,1.)
initweights=weights
#Trainingschessboards
chessboards,isChessboard=make_training_set(10,5)
optimal_weights=[]

print(chessboards)

#optimal_weights=train_nn(weights,chessboards,isChessboard)      #Aufrufen der Trainigsmethode (Für Aufgabe 3d)
#-----Code Für Aufgabe 3e)------------
x_achse=[]
errors=[]
weight_history=[]
optimal_weights,x_achse,errors,weight_history= train_nn_history(weights,chessboards,isChessboard)
weight_history=np.array(weight_history)
plt.plot(x_achse,errors)
plt.xlabel("Durchlauf")
plt.ylabel("Error")
plt.show()
print(weight_history)
plt.plot(x_achse,weight_history[:,0],label="Gewicht1")
plt.plot(x_achse,weight_history[:,1],label="Gewicht2")
plt.plot(x_achse,weight_history[:,2],label="Gewicht3")
plt.plot(x_achse,weight_history[:,3],label="Gewicht4")
plt.legend()
plt.xlabel("Durchlauf")
plt.ylabel("Gewicht")
plt.show()

#-----Ende Code für Aufgabe 3e)

quick_check(chessboards,optimal_weights,isChessboard)
print("Anfängliche Gewichte: ",initweights)
print("Die optimalen gefundenen Gewichte: ",optimal_weights)