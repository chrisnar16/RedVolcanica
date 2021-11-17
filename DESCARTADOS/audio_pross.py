from scipy.io import wavfile
import scipy.io


def get_signal(in_file):
    fs, y = wavfile.read(in_file)
    num_type = y[0].dtype
    if num_type == 'int16':
        y = y * (1.0 / 32768)
    elif num_type == 'int32':
        y = y * (1.0 / 2147483648)
    elif num_type == 'float32':
        # Nothing to do
        pass
    elif num_type == 'uint8':
        raise Exception('8-bit PCM is not supported.')
    else:
        raise Exception('Unknown format.')
    if y.ndim == 1:
        return y, fs
    else:
        return y.mean(axis=1), fs
