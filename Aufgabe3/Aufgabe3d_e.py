import Neurons as N
import numpy as np
import matplotlib.pyplot as plt
import random

#-----Definitionen nützlicher Funktoinen----------


#Neuronales Netz zur Überprüfung ob ein Schachbrett korrekt identifiziert/nicht identifiziert wird (1DArray,1DArray,Boolean)
def TestChessboard(Chessboard,weights,korrekt):

    Neurons = N.setupNetwork(Chessboard) # Erstellung und Anregung der Input-Neuronen nach Brettvorlage

    V_Features_Out = [Neurons[0].get_V_max(), Neurons[1].get_V_max(), Neurons[2].get_V_max(),
                      Neurons[3].get_V_max()]  # Sämtliche äußere Spannungsmaxima der Input Neuronen


    I_5=np.dot(V_Features_Out,weights)
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

#Überprüfung durch neuronale Netze, ob Boardset korrekt erkannt wurde
def network_check(Chessboards,weights,areChessboards):
    success = True
    for i in range(len(Chessboards)):  # Durchgehen aller Chessboard um zu überprüfen, ob einen falsch erkannt wurde
        if not TestChessboard(Chessboards[i], weights, areChessboards[i]):  # Falls, ein Board fehlerhaft erkannt wurde
            success = False
    if success:
        print("Test erfolgreich!")  # Rückgabe, ob alle Boards korrekt erkannt werden konnten
    else:
        print("Test nicht erfolgreich!")


def sigmoid(x): #Logistische Sigmoidfunkttion
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):      #Ableitung logistischer Sigmoidfunktion
    return sigmoid(x) * (1 - sigmoid(x))
def binary_cross_entropy(y_true, y_pred):#Cross entropy Funktion, zur Berechnung des losses, da binäre Ausgabe
    # Vermeidung von log(0) durch Clipping der Werte
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_derivative(y_true, y_pred):
    # Vermeidung von Division durch 0 durch Clipping der Werte
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

#Trainingsalgorythmus zur bestimmung der optimalen Gewichte
def train_nn(initial_weights, training_boards, targets, learning_rate=0.01, epochs=100):
    # Initialisieren der Gewichte
    weights = np.array(initial_weights)

    for epoch in range(epochs):
        total_loss = 0
        for board, target in zip(training_boards, targets):
            # Berechnung Aktivierung z
            z = np.dot(board, weights)
            prediction = sigmoid(z)

            # Berechne Binary Cross Entropy Loss
            loss = binary_cross_entropy(target, prediction)
            total_loss += loss

            # Rückwärtspropagation
            error_derivative = binary_cross_entropy_derivative(target, prediction)
            adjustments = error_derivative * sigmoid_derivative(z)

            # Gewichte anpassen
            weights -= learning_rate * adjustments * np.array(board)

    return weights



#Trainingsalgorythmus zur bestimmung der optimalen Gewichte und Plotten der zeitl. Entwicklung der Gewichte und Fehler
def train_nn_history(initial_weights, training_boards, targets, learning_rate=0.01, epochs=100):
    # Initialisiere die Gewichte

    #----Modifizierung zur Fehlerplottung----

    weights=np.copy(initial_weights)
    x_achse=np.linspace(1,epochs,epochs)
    weight_history=[]
    error_history=[]
    for epoch in range(epochs):
        total_loss = 0
        weight_history.append(np.copy(weights))
        errors=0             #Anzahl gemachter Fehler in dieser Runde
        for board, target in zip(training_boards, targets):     #Iteration über alle Boards des Trainingsets mit zugehörigem Wahrheitswert
            # Forward Pass
            z = np.dot(board, weights)
            #Vereinfachung der Vorhersage um ständiges Lösen des Neuronen GDL-Systems zu vermeiden
            prediction = sigmoid(z)

            # Berechne Binary Cross Entropy Loss
            loss = binary_cross_entropy(target, prediction)
            total_loss += loss
            #Überprüfung ob Fehler gemacht wurde (Wenn Loss obere Schwelle überschreitet)
            if loss>.5:
                errors+=1
            # Rückwärtspropagation durch binary cross Ableitung
            error_derivative = binary_cross_entropy_derivative(target, prediction)
            adjustments = error_derivative * sigmoid_derivative(z)

            # Gewichte anpassen
            weights -= learning_rate * adjustments * board


        error_history.append(errors)    #Speichern, wie viele Fehler in der Runde gemacht wurden




    return weights,x_achse,error_history,weight_history#Rückgabe aller zum Plotten relevanter Daten
def gen_False_board():     #Funktion zur zufälligen erzeugung eines Schachbretts mit rückgabe, ob es ei korrektes Schachbrett ist
    pattern = [0, 1, 1, 0]  # Wahres Feld
    while np.array_equal(pattern,[0,1,1,0]):
        np.random.shuffle(pattern)  #Verändert das Schachbrett, sodass es kein wahres Feld mehr ist aber noch 2 1eun und nullen besitzt
    return pattern



def make_training_set(size=100,true_boards=50):     #Funktion, welche ein Trainingsset (Chessboards und Wahrheitsliste) mit gewünschter Größe und Anteil wahrer Boards enthält
    chessboards=np.zeros((size,4))      #Initialisierung leerer Chessboards
    arechessboards=[]               #Initialisierung leerer Wahrheitsliste
    already_set=[]                      #Liste zur überprüfung ob Element aus chessboads bereits generiert wurde
    for i in range(true_boards):      #Initiales Auffüllen aller wahren boards an zufälligen stellen
        spot=random.randint(0,size-1)#Spot zum Auffüllen in chessboards
        while spot in already_set:      #Überpfrüfung, ob spot bereits platziert wurde, wenn ja dann suche neuen spot
            spot=random.randint(0,size-1)
        chessboards[spot]=[0,1,1,0]  #Platzierung von wahrem Board in spot
        already_set.append(spot)         #Markierung, dass chessboard an spot bereits aufgefüllt wurde
    still_left=[]                       #Orte die noch aufgefüllt werden müssen
    for i in range(size):
        if i not in already_set:         #Wenn in i noch nicht gesetzt wurde soll dies in liste noch zu setzender Felder hinzugefügt werden
            still_left.append(i)

    for i in still_left:      #Restliches Auffüllen aller falschen Boards
       chessboards[i]=gen_False_board()  #Platzierung von falschem Board

    for i in range(size):           #Erstellen der Wahrheitsliste ob Board auch Schachbrettmuster ist
        if np.array_equal(chessboards[i],[0,1,1,0]):
            arechessboards.append(True)
        else:
            arechessboards.append(False)

    return chessboards,arechessboards
#-------Ende Definitoinen-----------------------


#-------Beginn Aufgeführter Code----------------

weights=np.zeros(4,dtype=float)     #Initialiseren der Anfangsgewichte
for i in range(4):                          #Zufälliges Wählen der Anfangsgewichte mit Werten zw. 0 und 1
    weights[i] = random.uniform(0.,1.)
initweights=np.copy(weights)
# Erstellung der Trainingschessboards
chessboards,isChessboard=make_training_set(50,25)
optimal_weights=[]



#Plotten des Fehlers im Verlauf der Epochen
x_achse=[]
errors=[]
weight_history=[]
print("Training...")
#Trainieren des NN
optimal_weights,x_achse,errors,weight_history= train_nn_history(weights,chessboards,isChessboard)
#Umwandlung Weitght Daten für Plotten
weight_history=np.array(weight_history)
#Plotten der Gemachten Fehler in jeder Epoche
plt.plot(x_achse,errors)
plt.xlabel("Epoche")
plt.ylabel("error")
plt.show()
#Plotten der Gewichte im Verlauf der Epochen
plt.plot(x_achse,weight_history[:,0],label="Gewicht1")
plt.plot(x_achse,weight_history[:,1],label="Gewicht2")
plt.plot(x_achse,weight_history[:,2],label="Gewicht3")
plt.plot(x_achse,weight_history[:,3],label="Gewicht4")
plt.legend()
Text=f'Anfangsgewichte: 1:{initweights[0]:.2}\n 2:{initweights[1]:.2}\n 3:{initweights[2]:.2}\n 4:{initweights[3]:.2}'
plt.text(100,0,Text,ha='right', va='center')
plt.xlabel("Durchlauf")
plt.ylabel("Gewicht")
plt.show()
print("Anfängliche Gewichte: ",initweights)
print("Die optimalen gefundenen Gewichte: ",optimal_weights)

#Überprüfung, ob Alle Boards richtig erkannt wurden (zum Testen, dauert lange, da Überprüfung über Neuronensimulation)
#network_check(chessboards,optimal_weights,isChessboard)

