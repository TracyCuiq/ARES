import torch.nn as nn
import torch
import numpy as np
import math

import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.autograd import Variable
import datetime
from gen_models import Generator_Unet
from YednetSE import YedNetSE

from utils import imsave_singel, show_result, quantize
from pgd import *
from gen_att_models import define_G, get_scheduler, set_requires_grad, Encoder

models_path = './models/'
img_prob_train_path = './train_result/'
img_prob_eval_path = './eval_result/'

# custom weights initialization called on netG and netD

def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            #m.weight.data.normal_(0, 0.02)
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.ConvTranspose2d):
            #nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
            m.bias.requires_grad = False

def grad_inv(grad):
    return grad.neg()

class ARESGAN:
    def __init__(self, device, img_nc=1, lr=0.0001, payld=0.4, bilinear=False):

        self.device = device
        self.img_nc = img_nc
        self.lr = lr
        self.payld = payld
        self.bilinear = bilinear
        self.netG = Generator_Unet(self.img_nc, bilinear=self.bilinear).to(self.device)
        self.netDisc = YedNetSE().to(self.device)
        self.netGAtt = define_G(3, 1, ngf=256, netG='unet_256', z_dim=20).to(self.device)
        self.encoder = Encoder(20).to(self.device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)
        #pretrained_model = './model0203/netG_epoch_200.pth'
        #self.netDisc.load_state_dict(torch.load(pretrained_model), strict=False)
        
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.l1_loss = torch.nn.L1Loss().to(self.device)
        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), self.lr)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(), self.lr)
        self.optimizer_GAtt = torch.optim.Adam(self.netGAtt.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)

        if not os.path.exists(models_path):
            os.makedirs(models_path)
        if not os.path.exists(img_prob_train_path):
            os.makedirs(img_prob_train_path)
        if not os.path.exists(img_prob_eval_path):
            os.makedirs(img_prob_eval_path)

    def train_batch(self, cover, label_group, label_zeros, epoch, TANH_LAMBDA=60, D_LAMBDA=1, Payld_LAMBDA=1e-7):
        batch_size = cover.shape[0]
        cover_1 = cover[:batch_size//2]
        cover_2 = cover[batch_size//2:]
        label_zeros = np.zeros(cover_2.shape[0])
        #label_ones = np.ones(cover_2.shape[0])
        #cover_2_lb = torch.from_numpy(label_ones).long().to(self.device)
        cover_2_lb = torch.from_numpy(label_zeros).long().to(self.device)
        #torch.ones_like(pred_real, device=self.device)

        # optimize D
        for i in range(1):
            self.optimizer_D.zero_grad()
            with torch.no_grad():
                prob_pred = self.netG(Variable(cover_1))
            data_noise = np.random.rand(prob_pred.shape[0], prob_pred.shape[1], prob_pred.shape[2], prob_pred.shape[3])
            tensor_noise = torch.from_numpy(data_noise).float().to(self.device)
            modi_map = 0.5*(torch.tanh((prob_pred+2.*tensor_noise-2)*TANH_LAMBDA) - torch.tanh((prob_pred-2.*tensor_noise)*TANH_LAMBDA))/255.
            stego = (cover_1 + modi_map)
            stego = quantize(stego)

            # attack cover
            # cover_adv = attack_Linf_PGD_bin(cover_2, cover_2_lb, self.netDisc, self.criterion, steps=5, epsilon=(1./255.))
            # calculate the two-step gradients as inputs to the generator
            ##################################################
            cover_2.requires_grad_()
            loss_ = F.cross_entropy(self.netDisc(cover_2), cover_2_lb)
            grad = torch.autograd.grad(loss_, [cover_2])[0].detach()
            cover_2.requires_grad_(False)

            x_fgsm = torch.clamp(cover_2 + 1.0/255.0 * grad.sign(), 0.0, 1.0).detach()
            x_fgsm.requires_grad_()
            grad_fgsm = torch.autograd.grad(F.cross_entropy(self.netDisc(x_fgsm), cover_2_lb), [x_fgsm])[0].detach()
            x_fgsm.requires_grad_(False)

            rand_z = torch.rand(cover_2.size(0), 20, device='cuda') * 2. - 1.
            #print(cover_2.shape, grad.shape, grad_fgsm.shape)
            #pert = self.netGAtt(torch.cat([cover_2, grad, grad_fgsm], 1), z=rand_z).tanh() # for resnet
            pert = self.netGAtt(torch.cat([cover_2, grad, grad_fgsm], 1)).tanh() #for unet
            
            self.optimizer_GAtt.zero_grad()
            self.optimizer_encoder.zero_grad()

            logits_z = self.encoder(pert)
            mean_z, var_z = logits_z[:, :batch_size], F.softplus(logits_z[:, batch_size:])
            neg_entropy_ub = -(-((rand_z - mean_z) ** 2) / (2 * var_z+1e-8) - (var_z+1e-8).log()/2. - math.log(math.sqrt(2 * math.pi))).mean(1).mean(0)

            cover_adv = torch.clamp(cover_2 + (1/255) * torch.clamp(pert, -1, 1), 0.0, 1.0)
            cover_adv.register_hook(grad_inv)
            loss_modi = self.l1_loss(cover_adv, cover_2) - self.payld##
            ################################################
            stego_ = torch.cat([stego, cover_adv])
            #data = torch.stack((cover_adv, stego))#
            #data = torch.stack((cover, stego))
            data = torch.stack((cover, stego_))

            data_shape = list(data.size())
            data = data.reshape(data_shape[0] * data_shape[1], *data_shape[2:])

            data_group = data.to(self.device)

            pred_D = self.netDisc(data_group.detach())
            loss_D = self.criterion(pred_D, label_group)
            loss_D += 1 * F.relu(neg_entropy_ub - 0.9)##
            loss_D += loss_modi##
            loss_D.backward()
            self.optimizer_D.step()
            self.optimizer_GAtt.step()
            self.optimizer_encoder.step()
  
        # optimize G
        for i in range(1):# update G 1 time
            img_size = cover.shape[2]
            batch_size = cover.shape[0]
            self.optimizer_G.zero_grad()

            prob_pred = self.netG(Variable(cover))
            data_noise = np.random.rand(prob_pred.shape[0], prob_pred.shape[1], prob_pred.shape[2], prob_pred.shape[3])
            tensor_noise = torch.from_numpy(data_noise).float().to(self.device)
            modi_map = 0.5*(torch.tanh((prob_pred+2.*tensor_noise-2)*TANH_LAMBDA) - torch.tanh((prob_pred-2.*tensor_noise)*TANH_LAMBDA))/255.
            stego = (cover + modi_map)
            torch.clamp(stego, 0.0, 1.0)

            data = torch.stack((cover, stego)) #data = torch.stack((cover, stego))
            data_shape = list(data.size())
            data = data.reshape(data_shape[0] * data_shape[1], *data_shape[2:])
            data_group = data.to(self.device)
            pred_D = self.netDisc(data_group)
            loss_D = self.criterion(pred_D, label_group)

            # cal G's loss in GAN
            prob_chanP = prob_pred / 2.0 + 1e-5
            prob_chanM = prob_pred / 2.0 + 1e-5
            prob_unchan = 1 - prob_pred + 1e-5

            cap_entropy = torch.sum( (-prob_chanP * torch.log2(prob_chanP)
                -prob_chanM * torch.log2(prob_chanM)
                -prob_unchan * torch.log2(prob_unchan) ),
                dim=(1, 2, 3)
                )

            payld_gen = torch.sum((cap_entropy), dim=0) / (img_size * img_size * batch_size)
            cap = img_size * img_size * self.payld
            loss_entropy = torch.mean(torch.pow(cap_entropy - cap, 2), dim=0)

            loss_G = D_LAMBDA * (-loss_D) + Payld_LAMBDA * loss_entropy
            loss_G.backward()
            self.optimizer_G.step()

    
        #return loss_D.data[0], loss_G.data[0]
        return loss_D.data.item(), loss_G.data.item()


    def train(self, train_dataloader, epochs):
        data_iter = iter(train_dataloader)
        sample_batch = next(data_iter)
        data_fixed = sample_batch['img'][0:]
        data_fixed = Variable(data_fixed.cuda())
        noise_fixed = np.random.rand(data_fixed.shape[0], data_fixed.shape[1], data_fixed.shape[2], data_fixed.shape[3])
        noise_fixed = torch.from_numpy(noise_fixed).float().to(self.device)
        noise_fixed = Variable(noise_fixed.cuda())

        label_zeros = np.zeros( data_fixed.shape[0] )
        label_ones = np.ones( data_fixed.shape[0] )
        label = np.stack((label_ones, label_zeros))
        label = torch.from_numpy(label).long()
        label = Variable(label).to(self.device)
        label_group = label.view(-1)
        label_zeros = torch.from_numpy(label_zeros).long().to(self.device)

        for epoch in range(1, epochs+1):
            loss_D_sum = 0
            loss_G_sum = 0
            
            for i, data in enumerate(train_dataloader, start=0):
                images = data['img']
                images = images.to(self.device)
                
                loss_D_batch, loss_G_batch = self.train_batch(images, label_group, label_zeros, epoch)
                loss_D_sum += loss_D_batch
                loss_G_sum += loss_G_batch
            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.6f, loss_G: %.6f" %
                  (epoch, loss_D_sum/num_batch, loss_G_sum/num_batch))
            # save generator
            if epoch%1==0:
                with torch.no_grad():
                    modi_map_fixed = 0.5*(torch.tanh((self.netG(data_fixed)+2.*noise_fixed-2)*60) - torch.tanh((self.netG(data_fixed)-2.*noise_fixed)*60))
                    stego_fixed = (data_fixed*255 + modi_map_fixed)/255.
                    #pert = self.netGAtt(torch.cat([cover_2, grad, grad_fgsm], 1)).tanh()
                show_result(epoch, self.netG(data_fixed), save=True, path=img_prob_train_path+str(epoch)+'prob.png')
                show_result(epoch, stego_fixed, save=True, path=img_prob_train_path+str(epoch)+'steg.png')
                show_result(epoch, modi_map_fixed, save=True, path=img_prob_train_path+str(epoch)+'modi.png')
                show_result(epoch, data_fixed, save=True, path=img_prob_train_path+str(epoch)+'cover.png')
                
            if epoch%100==0:
                netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth'
                netD_file_name = models_path + 'netD_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)
                torch.save(self.netDisc.state_dict(), netD_file_name)
