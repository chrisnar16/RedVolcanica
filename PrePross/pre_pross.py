import json
import math

import pandas as pd
import random
import numpy as np
from scipy import signal
import hashlib

import h5py


def open_bdd(nombre):
    return pd.read_json(nombre)


# extrae señales de un tipo y quita NaN
def extraer_señales(base, tipo, keytype='Type', keysp='StartPoint', keyep='EndPoint'):
    base_fil = base
    if tipo != '':
        is_type = base[keytype] == tipo
        base_fil = base[is_type]
    # dropnna quita na en esos valores
    base_fil.dropna(subset=[keysp], inplace=True)
    base_fil.dropna(subset=[keyep], inplace=True)
    base_fil.reset_index(drop=True, inplace=True)  # todo: revisar esto
    return base_fil


# todo: revisar codigo ya tenido(overlaping, ventana, euristica, tipo de verntana, )
# todo: normalizar tiempo en rms en amplitus tiempo

# todo: ver como ingresar datos a pytorech
# todo: transformar a db, espectrograma magnitud(cuidado con los 0) EPSILON PARA VALOR DE 0 (EN PYTORCH)
# todo: CREAR DIRECTORIOS, VER COMO CARGA LABEL
# TODO: ADAPTAR CODIGO A BASE DE DATOS

# TODO: VER DATOS EN OTRAS CATEGORIAS

# TODO: MANDARLE CODIGO A PAOLA


def find_max_size(base, keyduration='Duration'):
    column = base[keyduration]
    return column.max()


# con interepolacion lineal
def normalize_samplerate_100(base, keysample='SampleRate', keydata='Data',
                             keyduration='Duration'):  # TODO: Alternativas a interpolacion lineal
    for i in range(base.shape[0]):
        if base[keysample][i] == 50:
            muestra = base[keydata][i]
            time = np.linspace(0., base[keyduration][i], len(muestra) * 2)
            y0 = muestra[0] / 2
            muestra.insert(0, y0)
            for j in range(1, len(muestra) - 1):
                y0 = muestra[2 * j - 1]
                y1 = muestra[2 * j]
                x0 = time[2 * j - 1]
                x = time[2 * j]
                x1 = time[2 * j + 1]
                y = y0 + (((y1 - y0) * (x - x0)) / (x1 - x0))
                muestra.insert(2 * j, y)
            base[keydata][i] = muestra
            base[keysample][i] = 100
            base[keyduration][i] = base[keyduration][i] * 2


# Va de la mano con normalizar en segundos, solo recorta la señal de 83 segundos a numero de muestras
def normalizar_muestras(base, numero_muestras, keydata='Data', keyduration='Duration', keyep='EndPoint'):
    for i in range(base.shape[0]):
        muestra = base[keydata][i]
        muestra_r = muestra[0:int(numero_muestras) - 1]
        base[keydata][i] = muestra_r
        base[keyduration][i] = numero_muestras
        fin = int(base[keyep][i])
        if fin > numero_muestras:
            print('Here NM')
            base[keyep][i] = numero_muestras - 1


# normalisa tamanio en segundos
def normalizar_tamanio_base(base, numero_segundos, keydata='Data', keysp='StartPoint', keyep='EndPoint',
                            keysr='SampleRate', keyduration='Duration'):
    for i in range(base.shape[0]):
        muestra = base[keydata][i]
        objetivo = numero_segundos * base[keysr][i]
        inicio = int(base[keysp][i])
        fin = int(base[keyep][i])
        evento_ori = muestra[inicio:fin]
        base[keysp][i] = 0
        tamanio_actual = fin - inicio
        # Caso: señal mayor a objetibo, recorto señal(evento incompleto)
        if tamanio_actual > objetivo:
            señal_ret = evento_ori[0:objetivo]  # TODO:Encontarr alternativa a este caso
            base[keyep][i] = objetivo - 1
            print('Here R')
        elif tamanio_actual < objetivo:
            evento_ori = muestra[inicio:]
            base[keyep][i] = fin - inicio - 1
            # si el evento es menor, pero la señal no, cojo toda la muestra y añado el ruido de la señal del final
            if len(evento_ori) > objetivo:
                señal_ret = evento_ori[0:objetivo]
            # si no relleno con ruido
            elif len(evento_ori) < objetivo:

                while len(evento_ori) < objetivo:
                    # nr = random.randint(fin, len(muestra) - 1)
                    nr = random.randint(0, inicio - 1)
                    vr = muestra[nr]
                    evento_ori.append(vr)
                señal_ret = evento_ori
            # caso señal justa
            else:
                señal_ret = evento_ori
        else:
            # caso señal justa
            señal_ret = evento_ori
        base[keydata][i] = señal_ret
        base[keyduration][i] = objetivo


def quitar_dc(base, keydata='Data', keysp='StartPoint', keyep='EndPoint'):
    for i in range(base.shape[0]):
        muestra = np.array(base[keydata][i], dtype=np.float)
        inicio = int(base[keysp][i])
        fin = int(base[keyep][i])
        evento = muestra[inicio:fin]
        rms = np.mean(evento)
        evento = evento - rms
        muestra[inicio:fin] = evento
        ruido = muestra[fin+1:]
        rms_ruido = np.mean(ruido)
        ruido = ruido - rms_ruido
        muestra[fin+1:] = ruido
        base[keydata][i] = muestra
    return base


def guardar_base_npy(base, keydata='Data', keytype='Type'):
    dictionary = {}
    for i in range(base.shape[0]):
        numero = i
        muestra = base[keydata][numero]
        tipo = base[keytype][numero]
        muestra_arr_ori = np.array(muestra)
        samplerate = base['SampleRate'][numero]
        f_ori, t_ori, zxx_ori = signal.stft(muestra_arr_ori, fs=samplerate, padded=True)
        espectro_magnitud = np.abs(zxx_ori)
        raiz = "basedesglosada/"
        suma = 0
        for j in range(100):
            suma = suma+espectro_magnitud[j]
        str2hash = (str(suma)+tipo).encode()
        hashh = hashlib.md5(str2hash)
        hassh_ex = hashh.hexdigest()
        dirr = raiz + hassh_ex
        np.save(dirr, espectro_magnitud)
        dictionary[hassh_ex] = {tipo: numero}
    with open('data.json', 'w') as fp:
        json.dump(dictionary, fp, indent=4)
    return dictionary


def guardar_base_h5(base, dim1, dim2, keydata='Data', keytype='Type'):
    fileName = 'baseh5/data1.h5'
    numOfSamples = base.shape[0]
    with h5py.File(fileName, "w") as out:
        out.create_dataset("X_train", (numOfSamples, dim1, dim2), dtype='float64')
        out.create_dataset("Y_train", (numOfSamples, 1), dtype='u1')
    with h5py.File(fileName, "a") as out:
        for i in range(base.shape[0]):
            muestra = base[keydata][i]
            tipo = base[keytype][i]
            tag = -1
            if tipo == 'VT':
                tag = 0
            elif tipo == 'LP':
                tag = 1
            muestra_arr_ori = np.array(muestra)
            samplerate = base['SampleRate'][i]
            f_ori, t_ori, zxx_ori = signal.stft(muestra_arr_ori, fs=samplerate, padded=True)
            espectro_magnitud = np.abs(zxx_ori)
            #espectro_magnitud = np.expand_dims(espectro_magnitud, axis=-1)
            out['X_train'][i, ...] = espectro_magnitud
            out['Y_train'][i, ...] = tag
