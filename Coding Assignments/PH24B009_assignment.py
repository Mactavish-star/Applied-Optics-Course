import numpy as np
import matplotlib.pyplot as plt

beta= 1
omega_0= 20.0
fs= 200
t= np.linspace(0, 10, fs*10)

#timedomain_function
x_t= np.exp(-beta* t) * np.cos(omega_0* t)

x_omega= np.fft.fft(x_t)
freq = np.fft.fftfreq(len(x_t), d=1/fs)
omega = 2*np.pi*freq

plt.figure()
plt.plot(t,x_t)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Damped Oscillation")
plt.show()

plt.figure()
plt.plot(omega,np.abs(x_omega))
plt.xlabel("Angular Frequency (rad/s)")
plt.ylabel("Magnitude")
plt.title("Fourier Spectrum")
plt.show()