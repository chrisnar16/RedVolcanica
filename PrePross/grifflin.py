import numpy as np
from scipy import signal


def stft_r(data, samplerate, stft_args={}):
    f, t, Zxx = signal.stft(data, samplerate, **stft_args)
    return f, t, Zxx


def reconstruir_señal_generador(espectrograma_magnitud, iteraciones, samplerate):
    señal_aproximada = None
    for i in range(iteraciones):
        if señal_aproximada is None:
            espectrograma_angulo = np.random.randn(espectrograma_magnitud.shape[0], espectrograma_magnitud.shape[1])
        else:
            f, T, espectrograma_rec = signal.stft(señal_aproximada, fs=samplerate, padded=True)
            espectrograma_angulo = np.angle(espectrograma_rec)
        espectrograma_rec = espectrograma_magnitud * np.exp(1j * espectrograma_angulo)
        time, señal_aproximada=signal.istft(espectrograma_rec, fs=samplerate)
    return time, señal_aproximada
