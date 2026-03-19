import numpy as np
import matplotlib.pyplot as pt

# basic equation of damped harmonic oscillator - x(t) = Ae^-γt * cos(wt)

# parameters
A=2
γ=0.75
f=15
w=2*np.pi*f

t=np.linspace(0,5,2000)
dt = t[1]-t[0]

x=A* np.exp(-γ*t)*np.cos(w*t)
X=np.fft.fft(x)
freq = np.fft.fftfreq(len(t), dt)

mask= freq>0
freq = freq[mask]
X=np.abs(X[mask])

pt.plot(t, x)
pt.xlabel("Time (s)")
pt.ylabel("Amplitude")
pt.title("Time domain of Damped Harmonic Oscillation")
pt.show()

peak_index = np.argmax(X[1:]) + 1 
f_peak = freq[peak_index]
omega = 2 * np.pi * f_peak
print(f"Peak frequency f = {f_peak:.2f} Hz")
A_peak = X[peak_index]

pt.plot(freq, X)
pt.xlabel("Frequency (Hz)")
pt.ylabel("Amplitude")
pt.title("FFT of Damped Harmonic Oscillation")

pt.axvline(x=f_peak, linestyle='--')   # mark peak frequency
pt.scatter(f_peak, A_peak)             # mark the peak point

# print text on graph
pt.text(f_peak+1, A_peak,
        f"f = {f_peak:.2f} Hz\nA = {A_peak:.2f}")

pt.show()