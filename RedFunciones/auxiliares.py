import torch.nn.functional as F
import torch
from torch import nn


def get_one_hot_labels(labels, n_classes):
    return F.one_hot(labels, n_classes)


def combine_vectors(x, y):
    combined = torch.cat((x.float(), y.float()), dim=1)
    return combined


def get_input_dimensions(z_dim, mnist_shape, n_classes):
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