#%matplotlib widget
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

b = pi/180 #factor grados a radianes
c = 180/pi #radianes a grados

def t_ll(theta_i,n1,n2):
    theta_i *= b
    theta_t = arcsin((n1/n2)* sin(theta_i))
    return (2*sin(theta_t)*cos(theta_i))/(sin(theta_i + theta_t)*cos(theta_i-theta_t))

theta_i = linspace(0, 90, 91)

fig, ax = plt.subplots()
line, = plt.plot(theta_i, t_ll(theta_i,1, 2), lw=2)
ax.set_xlabel('Ángulo')
plt.subplots_adjust(left=0.25, bottom=0.35)

axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='n1',
    valmin=0.1,
    valmax=3,
    valinit=1,
)
axamp = plt.axes([0.25, 0.2, 0.65, 0.03])
amp_slider = Slider(
    ax=axamp,
    label='n2',
    valmin=0.1,
    valmax=3,
    valinit=2
)
def update(val):
    line.set_ydata(f(theta_i, freq_slider.val, amp_slider.val))
    fig.canvas.draw_idle()
    
freq_slider.on_changed(update)
amp_slider.on_changed(update)
plt.show()

