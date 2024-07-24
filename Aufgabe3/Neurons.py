import numpy as np
import Functions as func
from scipy.integrate import odeint

# Parameter zur Neuronensimulation
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


# Ergebnisse extrahieren (Syntax Info)
#V_values = solution[:, 0]
#n_values = solution[:, 1]
#m_values = solution[:, 2]
#h_values = solution[:, 3]

#Definieren der Klasse Neuron mit sämtlichen für das neuronale Netzwerk relevanten Funktionen
class Neuron:

    Field = False
    Solution = []
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
    def get_V_max(self):
        if max(self.Solution[:,0])<-5:#Verhindern, dass I<I0 vorkommt
            return -5
        return  max(self.Solution[:,0])



class TargetNeuron:
    I_ext_max=0
    Solution = []
    n0 = 0.3177  # Initialer Wert von n
    m0 = 0.0529  # Initialer Wert von m
    h0 = 0.5961  # Initialer Wert von h
    y0 = [V0, n0, m0, h0]
    def I_Input(self,t): #Funktion, welche im DGL System erlaubt, die effektive angelegte Spannung al Funktion zu nutzen
        return self.I_ext[int(t/dt)] #Gibt die angelegte Spannung zur jeweiligen Zeit als zugehöriges Array Element zurück



    def __init__(self,I_ext_max):#Konstruktor, welcher als einziges Attribut die bereits gewichtete Spannungskurve der vorgeschaltetetn Neuronen erhält
        self.I_ext_max = I_ext_max
        self.Solution = odeint(func.hodgkin_huxley, self.y0, t,args=(I_ext_max,))      #Speichern der Spannungskurven in das Objekt


    def get_activation(self):            #Ausgabe der eigenen Aktivierung
        return self.Solution[:,2]           # Gibt Aktivierung(m-Variable) zurück
def setupNetwork(chessboard):
    Neurons=[]
    for x in range(4):  # appending empty objects
        if chessboard[x] == 1:         #Initialiseirt die Input Neuronen mit entsprechenden Startspannungen, je nach Schwarz/Weiss
            Neurons.append(Neuron(True))   #Neuron Aktiviert
        else:
            Neurons.append(Neuron(False))  #Neuron Deaktiviert
    return Neurons
