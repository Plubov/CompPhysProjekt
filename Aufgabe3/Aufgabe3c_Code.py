import numpy as np
import matplotlib.pyplot as plt
import Functions as func
import Neurons as N

#Funktion zur Simulation des neuronalen Netzes zur Erkennung des Schachbrettmusters mit eingebauter Überpfüfung
def TestChessboard(Chessboard,weights,korrekt):

    Neurons = N.setupNetwork(Chessboard) # Erstellung und Anregung der Input-Neuronen

    V_Features_Out = [Neurons[0].get_V_max(), Neurons[1].get_V_max(), Neurons[2].get_V_max(),
                      Neurons[3].get_V_max()]  # Sämtliche äußere Spannungsmaxima der Input Neuronen

    I_5 = 0  # Letzendliche Äußere maximale Stromstärke an Output Neuron als Summe aller Spannungen der Feature Neuronen mult. mit ihren Gewichten

    for i in range(0, len(V_Features_Out)):  # Addiren aller Spannungen der Input Neuronen mult. mit ihrem Gewicht
        I_5 += weights[i] * V_Features_Out[i]

    if I_5 < -5:  # Verhinderung, dass I < I0 vorkommt
        I_5 = -5
    TargetNeuron = N.TargetNeuron(I_5)  #Initialisierung Target-Neuron mit endgültiger angelegter Spannung
    print("I_5: ",I_5)
    print("Maximale Aktivierung des Target neurons: ",max(TargetNeuron.get_activation()))  # Ausgabe der Aktivierung (Erkennungswahrscheinlichkeit für Schachbrett) des Target-Neuurons
    #Ausgabe ob NN korrekt funktionierte  mit jeweiliger Fallunterscheidung
    if max(TargetNeuron.get_activation())>=.5:
        if korrekt:
            print("Success!: Schachbrettmuster korrekt erkannt")
        else:
            print("Fail!: Schachbrettmuster fälschlicherweise erkannt")
    else:
        if korrekt:
            print("Fail! : Schachbrettmuster fälschlicherweise nicht erkannt")
        else:
            print("Success! : Schachbrettmuster korrekterweise nicht erkannt")

weights =[1,1,1,1]  #Gewichte der Input Neuronen
#Verschiedene Schachfelder (Schachmuster True/False)
Chessboard1=[0,1,1,0]# True
Chessboard2=[1,1,0,0]#False
Chessboard3=[1,0,0,1]#False
Chessboard4=[0,0,1,1]#False

#Testen der verschiedenen Schachbretter mit Uniformen Gewichten auf 1 aus 1
print("--------Versuche ohne optimierte Gewichte---------")
TestChessboard(Chessboard1,weights,True)
TestChessboard(Chessboard2,weights,False)
TestChessboard(Chessboard3,weights,False)
TestChessboard(Chessboard4,weights,False)

#Setzen der Gewichte auf optimale Werte
weights=[0.,.05,.05,0]
print("-------Versuche mit optimierten Gewichten--------")
print("Optimale Gewichte: ",weights)
#Testen der verschiedenen Schachbretter mit optimalen Gewichten
TestChessboard(Chessboard1,weights,True)
TestChessboard(Chessboard2,weights,False)
TestChessboard(Chessboard3,weights,False)
TestChessboard(Chessboard4,weights,False)




