import numpy as np
import Functions as func
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameter
Cm = 1.0  # Membrankapazität in uF/cm^2
gK = 36.0  # Maximale Leitfähigkeit für Kalium in mS/cm^2
gNa = 120.0  # Maximale Leitfähigkeit für Natrium in mS/cm^2
gL = 0.3  # Maximale Leitfähigkeit für Leckstrom in mS/cm^2
EK = -77.0  # Umkehrpotential für Kalium in mV
ENa = 50.0  # Umkehrpotential für Natrium in mV
EL = -54.387  # Umkehrpotential für Leckstrom in mV
I0 = 10.0  # Konstanter angelegter Strom in uA/cm^2

# Zeitparameter
dt = 0.01  # Zeitschritt in ms
t_max = 50.0  # Maximale Zeit in ms
t = np.linspace(0, t_max, int(t_max / dt))  # Zeitvektor

# Anfangsbedingungen
V0 = -65.0  # Initiales Membranpotential in mV
n0 = 0.3177  # Initialer Wert von n
m0 = 0.0529  # Initialer Wert von m
h0 = 0.5961  # Initialer Wert von h
y0 = [V0, n0, m0, h0]

# Lösung des Systems
solution = odeint(func.hodgkin_huxley, y0, t, args=(I0,))

# Ergebnisse extrahieren
V_values = solution[:, 0]
n_values = solution[:, 1]
m_values = solution[:, 2]
h_values = solution[:, 3]

#Definieren der Klasse Neuron mit sämtlichen für das neuronale Netzwerk relevanten Funktionen
class Neuron:

    V0=0         # Initialer Wert von I, wird im Konstruktor Überschrieben
    n = 0.3177  # Initialer Wert von n
    m = 0.0529  # Initialer Wert von m
    h = 0.5961  # Initialer Wert von h
    def __init__(self,V0):
        self.V0=V0



    def solve(self):
        y = [self.V0, self.n, self.m, self.h]
        solution = odeint(func.hodgkin_huxley, y, t, args=(I0,))
        self.V0 = solution[:, 0]
        self.n = solution[:, 1]
        self.m = solution[:, 2]
        self.h = solution[:, 3]
    def init_solve(self,y0):
        solution = odeint(func.hodgkin_huxley, y0, t, args=(I0,))
        self.V0 = solution[:, 0]
        self.n = solution[:, 1]
        self.m = solution[:, 2]
        self.h = solution[:, 3]

Neurons = [] #empty array
Chessboard=[1,0,0,1]# 1=Weiss, 0=Schwarz

for x in range(4):  # appending empty objects
    if Chessboard[x]==1:         #Initialiseirt die Input Neuronen mit entsprechenden Startspannungen, je nach Schwarz/Weiss
        Neurons.append(Neuron(15))   #Neuron Deaktiviert
    else:
        Neurons.append(Neuron(-5))  #Neuron Aktiviert




V_Features_Out =
weights =[1,1,1,1]
I_5=np.dot(weights,V_Features_Out) # Letzendliche Äußere Stromstärke an Output Neuron als Summe aller Spannungen der Feature Neuronen mult. mit ihren Gewichten




