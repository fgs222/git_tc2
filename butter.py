import math

import control
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from scipy import signal as sig


def get_epsilon(alpha_max, decimals):
    # Calcula el valor de epsilon
    # In: alfa maximo, cantidad de decimales
    # Out: epsilon
    eps = math.sqrt(10 ** (0.1 * alpha_max) - 1)
    eps = round(eps, decimals)
    print("Epsilon: " + str(eps))
    return eps


def get_bw_order(alpha_max, alpha_min, wp, ws):
    # Calcula el orden del butterworth para maxima planicidad
    # In: alfa maximo, alfa minimo, frecuencia angular de paso y stop
    # Out: orden del LPF
    num = math.log((10 ** (0.1 * alpha_min) - 1) / (10 ** (0.1 * alpha_max) - 1), 10)
    den = 2 * math.log(ws / wp, 10)
    order = math.ceil(num / den)
    print("Orden: " + str(order))
    return order


def make_den(order, eps, wp):
    # Crea un vector con los coeficientes del denominador segun el orden del filtro
    # In: order, epsilon
    # Out: vector coeficientes del denominador

    coef = []
    aux = 0

    while aux < (order * 2) + 1:
        if aux == 0:
            coef.append(1)
        elif aux < order * 2:
            coef.append(0)
        else:
            coef.append(1)
        aux += 1

    return coef


def get_tf_poles(poles_s):
    # Elimina los polos positivos
    # In: polos (en s)
    # Out: polos positivos (en s)
    poles = []

    for pole in poles_s:
        if pole.real <= 0:
            poles.append(pole)
    return poles


def get_qpole(poles):
    # Calcula el valor de Q asociado a los polos
    # In: poles
    # Out: Q
    values = []
    for pole in poles:
        angle = math.atan2(pole.imag, pole.real)
        Q = - round(1 / (2 * math.cos(angle)), 2)
        values.append(Q)
    return values


def plot_bode(tf, title):
    # Grafica el bode de la transferencia
    # In: transferencia, titulo
    # Out:
    w, mag, phase = sig.bode(tf)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w, y=mag))
    fig.update_layout(title=title, xaxis_title="w", yaxis_title="mag",
                    font=dict(family="Courier New, monospace", size=18),
                    xaxis_type="log", yaxis_type="linear")
    fig.show()


def get_so_lpf(Q, w0):
    # Crea una transferencia lpf de orden 2
    # In: Q, w0
    # Out: numerador y denominador de transferencia lpf orden 2
    num = w0 ** 2
    den = [1, w0 / Q, w0 ** 2]
    return num, den


def get_sk_values(Q, C):
    # Calcula los parametros de la topologia sallen key
    # In: Q, C
    # Out: R, Rb
    R = round(1 / C, 1)
    k = 3 - (1 / Q)
    Rb = round((k - 1) * R, 1)
    return R, Rb


# Alfas maximos y minimos, frecuencias de paso y stop
alfa_max = 1
alfa_min = 35
f_pass = 1000
f_stop = 3500
wp = 2 * math.pi * f_pass
ws = 2 * math.pi * f_stop

# Epsilon, orden del filtro y wb
epsilon = get_epsilon(alfa_max, 3)
orden = get_bw_order(alfa_max, alfa_min, wp, ws)
wb = wp * epsilon ** (-1 / orden)

# Cuadrado del modulo de la transferencia para maxima planicidad (en w)
num_sq = [1]
den_sq = make_den(orden, epsilon, wp)
print(den_sq)
tfw_sq = sig.TransferFunction(num_sq, den_sq)

polos_w = np.around(tfw_sq.poles, 2)
ceros_w = np.around(tfw_sq.zeros, 2)

# Cuadrado del modulo de la transferencia para maxima planicidad (en s)
# s = jw

polos_s = polos_w / 1j
ceros_s = ceros_w / 1j

tf_sq = sig.zpk2tf(ceros_s, polos_s, 1)

control.pzmap(control.TransferFunction(tf_sq[0], tf_sq[1]), Plot=True,
              title="Polos y ceros de |H(s)|^2")
plt.show()

# Transferencia del filtro deseado
# Elimino polos positivos

polos = np.sort(get_tf_poles(polos_s))
ceros = ceros_s

tf = sig.zpk2tf(ceros, polos, 1)
print('Cantidad de polos: ' + str(len(polos)))
print("Polos: " + str(polos))
print("Cantidad de ceros: " + str(len(ceros)))
print("Ceros: " + str(ceros))

control.pzmap(control.TransferFunction(tf[0], tf[1]), Plot=True,
              title="Polos y ceros de |H(s)|")
plt.show()

q_polos = get_qpole(polos)
print("Q de los polos: " + str(q_polos))
print("Coeficientes del numerador: " + str(tf[0]))
print("Coeficientes del denominador: " + str(tf[1]))

plot_bode(tf, "Respuesta en Frencuencia")

# Creo transferencias de orden 2

tf1 = get_so_lpf(abs(q_polos[0]), 1)
tf2 = get_so_lpf(abs(q_polos[2]), 1)

tf1 = sig.TransferFunction(tf1[0], tf1[1])
tf2 = sig.TransferFunction(tf2[0], tf2[1])

plot_bode(tf1, "Respuesta en Frencuencia tf1")
plot_bode(tf2, "Respuesta en Frencuencia tf2")

print("Coeficientes del numerador tf1: " + str(tf1.num))
print('Coeficientes del denominador tf1: ' + str(tf1.den))
print("Coeficientes del numerador tf2: " + str(tf2.num))
print('Coeficientes del denominador tf2: ' + str(tf2.den))

# Valores de los componentes de la topologia Sallen Key
C = 470E-06

R1, Rb1 = get_sk_values(q_polos[0], C)
R2, Rb2 = get_sk_values(q_polos[2], C)

print("R1: " + str(R1) + ", Rb1: " + str(Rb1) + (", C: " + str(C)))
print("R2: " + str(R2) + ", Rb2: " + str(Rb2) + (", C: " + str(C)))

# Convertir LPF a HPF
# Transformacion P = 1/S

thp1 = sig.TransferFunction([1, 0, 0], tf1.den)
thp2 = sig.TransferFunction([1, 0, 0], tf2.den)

plot_bode(thp1, "Respuesta en Frencuencia thp1")
plot_bode(thp2, "Respuesta en Frencuencia thp2")

print("Coeficientes del numerador thp: " + str(thp1.num))
print('Coeficientes del denominador thp1: ' + str(thp1.den))
print("Coeficientes del numerador thp2: " + str(thp2.num))
print('Coeficientes del denominador thp2: ' + str(thp2.den))
