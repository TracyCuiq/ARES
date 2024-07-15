import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
#from torch.utils.data.dataset import Dataset
from ARESGAN import ARESGAN
from gen_models import Generator_Unet
from Dataloader import MyDataset
from utils import imsave_singel, show_result
import numpy as np

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    kwargs = {'num_worker': 4, 'pin_memory': True}


    Train = True
    use_cuda = True
    img_nc = 1
    payload = 0.4
    epochs_train = 400
    epochs_eval = 1
    lr = 0.0001
    batch_size_train = 20
    batch_size_eval = 1
    bilinear = True
    DIR = '/mnt/ssd1/cuiqi/dataset/BOSSBase_256/'
    tstDIR = 'ALASKA_v2_TIFF_256_RD_40000_pgm/'

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    pretrained_model = './models0506/netG_epoch_150.pth'
    img_prob_eval_path = './eval_result/'
    img_cover_eval_path = './ALASKA_v2_TIFF_256_RD_10000_pgm_RobSte_Yed_ADT_ep200_4tr/'
    img_stego_eval_path = './ALASKA_v2_TIFF_256_RD_10000_pgm_RobSte_Yed_ADT_ep200_ste044tr/'

    mytransform = transforms.Compose([transforms.ToTensor(),]) 
    if Train:
        dataset = MyDataset(DIR, transform=mytransform )
        dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)
        GAN_trainer = ARESGAN(device, img_nc, lr, payload, bilinear)
        GAN_trainer.train(dataloader, epochs_train)
    else:
        dataset = MyDataset(tstDIR, transform=mytransform )
        dataloader = DataLoader(dataset, batch_size=batch_size_eval, shuffle=False, num_workers=1)
        model = Generator_Unet(img_nc, 1, bilinear)
        model.load_state_dict(torch.load(pretrained_model), strict=False)
        model.cuda()
        model.eval()
        for i, data in enumerate(dataloader, start=1):
            images = data['img']
            images = images.to(device)
            prob_pred = model(images)
            #imsave_singel(prob_pred, path=img_prob_eval_path + str(i) +'.tif')
            data_noise = np.random.rand(prob_pred.shape[0], prob_pred.shape[1], prob_pred.shape[2], prob_pred.shape[3])  
            tensor_noise = torch.from_numpy(data_noise).float().to(device)
            TANH_LAMBDA = 1000000
            modi_map = 0.5*(torch.tanh((prob_pred+2.*tensor_noise-2)*TANH_LAMBDA) - torch.tanh((prob_pred-2.*tensor_noise)*TANH_LAMBDA))
            stego = (images*255 + modi_map)
            stego[stego < 0] = 0
            stego[stego > 255] = 255
            stego = stego/255.
            # imsave_singel(images, path=img_cover_eval_path + str(i) + '.pgm')
            imsave_singel(stego, path=img_stego_eval_path + str(i) + '.pgm')


if __name__ == '__main__':
    main()