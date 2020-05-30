import control
import math
import matplotlib.pyplot as plt
from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import signal as sig


def plot_bode(tf):
    # Grafica el bode de la transferencia
    # In: transferencia
    # Out:
    w, mag, phase = sig.bode(tf)

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Respuesta en Frecuencia', 'Fase'))
    fig.add_trace(go.Scatter(x=w, y=mag), row=1, col=1)
    fig.add_trace(go.Scatter(x=w, y=phase), row=2, col=1)

    fig['layout']['xaxis']['title'] = 'w'
    fig['layout']['xaxis2']['title'] = 'w'
    fig['layout']['yaxis']['title'] = 'mag'
    fig['layout']['yaxis2']['title'] = 'phase'

    fig.update_layout(title="Respuesta en frecuencia y fase",
                      font=dict(family="Courier New, monospace", size=18),
                      xaxis_type="log", yaxis_type="linear")

    fig.show()

    plot(fig, filename='5_21_tf.html', auto_open=False)

Q = 2.22
k = 1 / 9

a = 4
b = 3.95
c = 4.95
d = 1.96

num = [1, 0, k, 0]
den = [1, 2, 2, 1]

tf = sig.TransferFunction(num, den)

print(tf.zeros)
print(tf.poles)

control.pzmap(control.TransferFunction(num, den), Plot=True,
              title="Polos y ceros de H(s)")
plt.show()

plot_bode(tf)
