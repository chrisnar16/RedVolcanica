from torch import nn
import torch


class Discriminator(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential( # 129, 33
            self.make_disc_block(im_chan, hidden_dim, kernel_size=(13, 4), stride=(1, 1)),# 59, 30
            self.make_disc_block(hidden_dim, hidden_dim * 2, kernel_size=(13, 4), stride=(2, 2)), # 53, 14
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4, kernel_size=(13, 4), stride=(1, 1)),# 41, 11
            self.make_disc_block(hidden_dim * 4, hidden_dim * 8, kernel_size=(13, 4), stride=(1, 1)),# 29, 8
            self.make_disc_block(hidden_dim * 8, hidden_dim * 16, kernel_size=(13, 4), stride=(1, 1)),# 17, 5
            self.make_disc_block(hidden_dim * 16, hidden_dim * 32, kernel_size=(13, 4), stride=(1, 1)),# 5, 2
            self.make_disc_block(hidden_dim * 32, 1, kernel_size=(5, 2), stride=(2, 2), final_layer=True),# 1, 1
            
            #self.make_disc_block(im_chan, hidden_dim, kernel_size=(13, 4), stride=(2, 2)),# 59, 15
            #self.make_disc_block(hidden_dim, hidden_dim * 2, kernel_size=(13, 4), stride=(2, 2)), # 24, 6
            #self.make_disc_block(hidden_dim * 2, hidden_dim * 4, kernel_size=(13, 4), stride=(2, 2)),# 6, 2
            #self.make_disc_block(hidden_dim * 4, 1, kernel_size=(6, 2), stride=(2, 2), final_layer=True),# 1, 1
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                #nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)