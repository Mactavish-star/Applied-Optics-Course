import numpy
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.widgets import Slider

#time axis
t = numpy.linspace(0, 1000, 10000)
freq = numpy.linspace(0,0.2,1000)

# These are in SI units, mass in Kg
m = 3
k = 3e-1
b = 0.05

# Initial Conditions x (m) and dx (m/s)
x_in = 5e-2
dx_in = 2.3e-2

# You define a model where the parameters are
"""
Z = [x, dx] - it is a list, plus it has the variables x and dx
t = time array - defined as a numpy array using the linspace function so that to define the time axis consideration for the oscillator
typically here the odeint will solve the homogenous Differential equation of the form
d^x + p*dx + q*x = 0
Hence we define:
p = (b/m)
q = (k/m)
"""

p = (b/m)
q = (k/m)

initial_condition = [x_in, dx_in]

# Defining the initial conditions is important hence we define it in the form of [x, dx] with some values

def differential_model(z, t, p, q):
    # we need to define dx/dt and d^2(x)/d(t^2) and return as a list
    dxdt = z[1]
    ddxdt2 = -(p*z[1] + q*z[0])
    return [dxdt, ddxdt2]

solution = odeint(differential_model, initial_condition, t, args=(p, q))

#Solution has two columns, one is for x, and other for dx, we need only the x for plotting.
x_solution = solution[:, 0]

# Number of samples
N = len(t)

# Time step
dt = t[1] - t[0]

# FFT computation
X = numpy.fft.fft(x_solution)

# Frequency axis
freq = numpy.fft.fftfreq(N, dt)

# Only keep positive frequencies
positive = freq > 0
freq = freq[positive]
X = X[positive]

# Amplitude spectrum
amplitude = numpy.abs(X)

# Restrict to 0–0.1 Hz
limit = freq <= 0.1

freq_zoom = freq[limit]
amplitude_zoom = amplitude[limit]

plt_2, ax = plt.subplots(2,1, figsize=(10,6))
plt_2.subplots_adjust(bottom=0.3)

#slider placement
slider_ax1 = plt_2.add_axes([0.2, 0.18, 0.6, 0.03])
slider_ax2 = plt_2.add_axes([0.2, 0.13, 0.6, 0.03])
slider_ax3 = plt_2.add_axes([0.2, 0.08, 0.6, 0.03])

# Slider definition
mass_slider = Slider(
    ax=slider_ax1,
    label="Mass",
    valmin=0.1,
    valmax=6,
    valinit=m
)

spring_constant_slider = Slider(
    ax=slider_ax2,
    label="Spring Constant (k)",
    valmin=0.01,
    valmax=1,
    valinit=k
)

dampen_constant_slider = Slider(
    ax=slider_ax3,
    label="Dampener(b)",
    valmin=0.001,
    valmax=0.7,
    valinit=b
)


#Time Domain Plts
line_t, = ax[0].plot(t, x_solution, label='Displacement (in m)')
ax[0].set_xlabel('Time (t)')
ax[0].set_ylabel('Displacement (in m)')
ax[0].set_title('Damped Oscillation Curve of Displacement-Time')
ax[0].legend()
ax[0].grid(True)

#Frequency Domain Plts
line_f, = ax[1].plot(freq_zoom, amplitude_zoom, label='Amplitude')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Amplitude')
ax[1].set_title('Fourier Spectrum (0–0.1 Hz)')
ax[1].set_xlim(0, 0.1)
ax[1].legend()
ax[1].grid(True)

def update(val):

    m = mass_slider.val
    k = spring_constant_slider.val
    b = dampen_constant_slider.val

    # Solve ODE
    x_data = odeint(differential_model, initial_condition, t, args=((b/m),(k/m)))[:,0]

    # Update time-domain plot
    line_t.set_ydata(x_data)

    # FFT
    new_X = numpy.fft.fft(x_data)

    # keep positive frequencies
    new_X = new_X[positive]

    # amplitude
    amplitude = numpy.abs(new_X)

    # restrict to 0–0.1 Hz
    amplitude_zoom_new = amplitude[limit]

    # update spectrum
    line_f.set_ydata(amplitude_zoom_new)

    plt_2.canvas.draw_idle()

mass_slider.on_changed(update)
spring_constant_slider.on_changed(update)
dampen_constant_slider.on_changed(update)
plt.tight_layout(rect=[0, 0.25, 1, 1])
plt.show()
