import torch.nn.functional as F
import torch
from torch import nn


def get_one_hot_labels(labels, n_classes):
    """
    Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
    Parameters:
        labels: tensor of labels from the dataloader, size (?)
        n_classes: the total number of classes in the dataset, an integer scalar
    """
    return F.one_hot(labels, n_classes)


def combine_vectors(x, y):
    """
    Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
    Parameters:
      x: (n_samples, ?) the first vector.
        In this assignment, this will be the noise vector of shape (n_samples, z_dim),
        but you shouldn't need to know the second dimension's size.
      y: (n_samples, ?) the second vector.
        Once again, in this assignment this will be the one-hot class vector
        with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
    """
    # Note: Make sure this function outputs a float no matter what inputs it receives
    combined = torch.cat((x.float(), y.float()), dim=1)
    return combined


# todo: adaptar para no minst
def get_input_dimensions(z_dim, mnist_shape, n_classes):
    """
    Function for getting the size of the conditional input dimensions
    from z_dim, the image shape, and number of classes.
    Parameters:
        z_dim: the dimension of the noise vector, a scalar
        mnist_shape: the shape of each MNIST image as (C, W, H), which is (1, 28, 28)
        n_classes: the total number of classes in the dataset, an integer scalar
                (10 for MNIST)
    Returns:
        generator_input_dim: the input dimensionality of the conditional generator,
                          which takes the noise and class vectors
        discriminator_im_chan: the number of input channels to the discriminator
                            (e.g. C x 28 x 28 for MNIST)
    """
    C, W, H = mnist_shape[0], mnist_shape[1], mnist_shape[2]
    generator_input_dim = z_dim + n_classes;
    discriminator_im_chan = C + n_classes;
    return generator_input_dim, discriminator_im_chan


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def get_noise(n_samples, input_dim, device='cpu'):
    """
    Function for creating noise vectors: Given the dimensions (n_samples, input_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        input_dim: the dimension of the input vector, a scalar
        device: the device type
    """
    return torch.randn(n_samples, input_dim, device=device)


# def interpolate_class(first_number, second_number):
#     first_label = get_one_hot_labels(torch.Tensor([first_number]).long(), n_classes)
#     second_label = get_one_hot_labels(torch.Tensor([second_number]).long(), n_classes)
#     # Calculate the interpolation vector between the two labels
#     percent_second_label = torch.linspace(0, 1, n_interpolation)[:, None]
#     interpolation_labels = first_label * (1 - percent_second_label) + second_label * percent_second_label
#     print(interpolation_labels)
#     # Combine the noise and the labels
#     noise_and_labels = combine_vectors(interpolation_noise, interpolation_labels.to(device))
#     fake = gen(noise_and_labels)
#     show_tensor_images(fake, num_images=n_interpolation, nrow=int(math.sqrt(n_interpolation)), show=False)