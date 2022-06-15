from math import floor


def calcular_hout_decom2d(Hin, kernel, stride):
    Hout = (Hin - 1)*stride+(kernel - 1) + 1
    return Hout


def calcular_stride_decom2d(Hout, Hin, kernel):
    stride = (Hout - 1 - (kernel - 1))/(Hin - 1)
    return stride


def calcular_kernel_decom2d(Hout, Hin, stride):
    kernel = ((Hout - 1) - ((Hin - 1)*stride)) + 1
    return kernel


def calcular_hout_com2d(Hin, kernel, stride):
    Hout = floor(((Hin - (kernel - 1) - 1)/(stride)) + 1)
    return Hout


def calcular_stride_com2d(Hout, Hin, kernel):
    stride = floor(((Hin + - (kernel - 1) - 1)/(Hout - 1)))
    return stride


def calcular_kernel_com2d(Hout, Hin, stride):
    kernel = -(((Hout - 1) * stride) + 1 - Hin) + 1
    return kernel
