from datetime import date
import torch
import torch.nn as nn
import torch.optim as optim
import os

cwd = os.getcwd()
DIRECTORY = cwd + '/checkpoints/'


def save_weighs(gen, disc, gen_opt, disc_opt, epoch, gen_loss, dis_loss, nombre):
    today = date.today()
    file_name = nombre + '-' + str(today) + '.pt'
    compelto = DIRECTORY + file_name
    file = open(compelto,'w+')
    file.close()
    torch.save({
        'gen':gen.state_dict(),
        'disc':disc.state_dict(),
        'gen_opt': gen_opt.state_dict(),
        'disc_opt': disc_opt.state_dict(),
        'epoch': epoch,
        'gen_loss': gen_loss,
        'dis_loss': dis_loss,
    }, compelto)
        