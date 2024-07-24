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


# Zeitparameter
dt = 0.01  # Zeitschritt in ms
t_max = 50.0  # Maximale Zeit in ms
t = np.linspace(0, t_max, int(t_max / dt))  # Zeitvektor


V0 = -65.0  # Initiales Membranpotential in mV
# Anfangsbedingungen Infos
#n0 = 0.3177  # Initialer Wert von n
#m0 = 0.0529  # Initialer Wert von m
#h0 = 0.5961  # Initialer Wert von h
#y0 = [V0, n0, m0, h0]


# Ergebnisse extrahieren Syntax Info
#V_values = solution[:, 0]
#n_values = solution[:, 1]
#m_values = solution[:, 2]
#h_values = solution[:, 3]

#Definieren der Klasse Neuron mit sämtlichen für das neuronale Netzwerk relevanten Funktionen
class Neuron:

    Field = False
    Solution=[]
    n0 = 0.3177  # Initialer Wert von n
    m0 = 0.0529  # Initialer Wert von m
    h0 = 0.5961  # Initialer Wert von h
    y0 = [V0, n0, m0, h0]
    def __init__(self,Field):#Konstruktor, welcher Übergibt, ob das Neuron Schwarz/False, Weiss/True ist
        if Field:
            self.Field = True
        self.Solution = odeint(func.hodgkin_huxley_dynamic, self.y0, t, args=(self.Field,))

    def get_V_Out(self):
        return self.Solution[:, 0]

class TargetN:
    I_ext_max=0
    Solution = []
    n0 = 0.3177  # Initialer Wert von n
    m0 = 0.0529  # Initialer Wert von m
    h0 = 0.5961  # Initialer Wert von h
    y0 = [V0, n0, m0, h0]
    def I_Input(self,t): #Funktion, welche im DGL System erlaubt, die effektive angelegte Spannung al Funktion zu nutzen
        return self.I_ext[int(t/dt)] #Gibt die angelegte Spannung zur jeweiligen Zeit als zugehöriges Array Element zurück

    def hodgkin_huxley_Feature(self,y, t):  # DGL System, welches das Feature Neuron mit vorgeschalteten strömen simuuliert
        I = self.I_ext_max         # Maximum der gewichteten vorgeschalteten Spannung
        V, n, m, h = y

        I_Na = gNa * m ** 3 * h * (V - ENa)
        I_K = gK * n ** 4 * (V - EK)
        I_L = gL * (V - EL)

        dVdt = (I - I_Na - I_K - I_L) / Cm
        dndt = func.alpha_n(V) * (1 - n) - func.beta_n(V) * n
        dmdt = func.alpha_m(V) * (1 - m) - func.beta_m(V) * m
        dhdt = func.alpha_h(V) * (1 - h) - func.beta_h(V) * h

        return [dVdt, dndt, dmdt, dhdt]

    def __init__(self,I_ext_max):#Konstruktor, welcher als einziges Attribut die bereits gewichtete Spannungskurve der vorgeschaltetetn Neuronen erhält
        self.I_ext_max = I_ext_max
        self.Solution = odeint(self.hodgkin_huxley_Feature, self.y0, t)      #Speichern der Spannungskurven in das Objekt


    def get_activation(self):            #Ausgabe der eigenen Spannungskurve
        return self.Solution[:,2]           # Gibt Aktivation (n-Variable) zurück


Neurons = [] #empty array
Chessboard=[0,1,1,0]# 1=Weiss, 0=Schwarz

for x in range(4):  # appending empty objects
    if Chessboard[x]==1:         #Initialiseirt die Input Neuronen mit entsprechenden Startspannungen, je nach Schwarz/Weiss
        Neurons.append(Neuron(True))   #Neuron Aktiviert
    else:
        Neurons.append(Neuron(False))  #Neuron Deaktiviert

plt.plot(t,Neurons[0].get_V_Out(),label='Neuron1')  #Plotten der einzelnen Input-Neuronen Spannungs-Signale
plt.plot(t,Neurons[1].get_V_Out(),label='Neuron2')
plt.plot(t,Neurons[2].get_V_Out(),label='Neuron3')
plt.plot(t,Neurons[3].get_V_Out(),label='Neuron4')
plt.xlabel("t in ms")
plt.ylabel("V in mV")
plt.legend()

V_Features_Out = [max(Neurons[0].get_V_Out()),max(Neurons[1].get_V_Out()),max(Neurons[2].get_V_Out()),max(Neurons[3].get_V_Out())]#Sämtliche äußere Spannungsmaxima der Input Neuronen
weights =[0,1,1,0]  #Gewichte der Input Neuronen
I_5=0 # Letzendliche Äußere maximale Stromstärke an Output Neuron als Summe aller Spannungen der Feature Neuronen mult. mit ihren Gewichten


for i in range(0,len(V_Features_Out)): # Addiren aller Spannungen der Input Neuronen mult. mit ihrem Gewicht
    I_5 += weights[i]*V_Features_Out[i]

if I_5<-5:      #Verhinderung, dass I < I0 vorkommt
    I_5=-5
TargetNeuron = TargetN(I_5)     # Erstellen der Target Neurons mit der Übergabe der zuvor berechneten maximalen gewichteten Anregungsspannung



plt.show()
print("Maximale Aktivierung des Target neurons: ",  max(TargetNeuron.get_activation())) #Ausgabe der Aktivierung (Erkennungswahrscheinlichkeit für Schachbrett) des Target-Neuurons
