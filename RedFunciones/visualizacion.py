from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import PrePross.grifflin as grifflin
import numpy as np

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    #image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()

        
def show_time_domine_images(image_tensor, std, mean, real,  num_images=25, size=(1, 28, 28), nrow=5, show=True):
    image_tensor = image_tensor * std + mean
    image_unflat = image_tensor.cpu().detach().numpy()
    samplerate = 50
    
    x = np.squeeze(image_unflat[2])
    x = np.transpose(x)
    timee, muestra_rec=grifflin.reconstruir_señal_generador(x, 100, samplerate)
    tamaño = len(muestra_rec) / samplerate
    time = np.linspace(0., tamaño, len(muestra_rec))
    plt.plot(time,muestra_rec)
    if real:
        plt.title("muestra real")
    else:        
        plt.title("muestra generada")     
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()
    #for i in range(1):
    #    x = np.squeeze(image_unflat[2])
    #    x = np.transpose(x)
    #    timee, muestra_rec=grifflin.reconstruir_señal_generador(x, 100, samplerate)
    #    tamaño = len(muestra_rec) / samplerate
    #    time = np.linspace(0., tamaño, len(muestra_rec))
    #    plt.plot(time,muestra_rec)
    #    plt.title("Señal Recuperada")
    #    plt.xlabel("Time [s]")
    #    plt.ylabel("Amplitude")
    #    plt.show()
    