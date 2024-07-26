import numpy as np
import Functions as func
import Neurons as N
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#Korrektes Chessboard (Wie in Dokumentation)
Chessboard=[0,1,1,0]# 1=Weiss, 0=Schwarz

Neurons= N.setupNetwork(Chessboard)#Erstellt Neuronennetz, welche durch Schachfelder aktiviert werden

plt.plot(N.t,Neurons[0].get_V_Out(),label='Neuron1')  #Plotten der einzelnen Input-Neuronen Spannungs-Signale (ohne Berücksichtigung der Signalverzögerung)
plt.plot(N.t,Neurons[1].get_V_Out(),label='Neuron2')
plt.plot(N.t,Neurons[2].get_V_Out(),label='Neuron3')
plt.plot(N.t,Neurons[3].get_V_Out(),label='Neuron4')
plt.xlabel("t in ms")
plt.ylabel("V in mV")
plt.legend()

V_Features_Out = [Neurons[0].get_V_max(),Neurons[1].get_V_max(),Neurons[2].get_V_max(),Neurons[3].get_V_max()]#Sämtliche äußere Spannungsmaxima der Input Neuronen
weights =[1,1,1,1]  #Gewichte der Input Neuronen
I_5=0 # Letzendliche Äußere maximale Stromstärke an Output Neuron als Summe aller Spannungen der Feature Neuronen mult. mit ihren Gewichten


for i in range(0,len(V_Features_Out)): # Addiren aller Spannungen der Input Neuronen mult. mit ihrem Gewicht
    print("V_out[",i,"]=",V_Features_Out[i])
    I_5 += weights[i]*V_Features_Out[i]
if I_5 < -5:  # Verhinderung, dass I < I0 vorkommt
    I_5 = -5

print("I_5=",I_5)
TargetNeuron = N.TargetNeuron(I_5)     # Erstellen der Target Neurons mit der Übergabe der zuvor berechneten maximalen gewichteten Anregungsspannung
print("V_Out_Target Neuron: ",TargetNeuron.get_V_max())

plt.plot(N.t,TargetNeuron.get_V_Out(),label='Target')
plt.show()
print("Maximale Aktivierung des Target neurons: ",  max(TargetNeuron.get_activation())) #Ausgabe der Aktivierung (Erkennungswahrscheinlichkeit für Schachbrett) des Target-Neuurons
