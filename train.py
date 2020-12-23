import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torch.nn import Conv2d,LeakyReLU,BatchNorm2d, ConvTranspose2d,ReLU
import cv2,datetime,os
from net import GeneratorNet,DiscrimiterNet
import torch.optim as optim
from dataset import MYDataSet
from utils import loss_igdl
import argparse
from tensorboardX import SummaryWriter
import numpy as np

def ToTensor(image):
    """Convert ndarrays in sample to Tensors."""
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    # Normalize image from [0, 255] to [0, 1]
    image = 1 / 255.0 * image
    return torch.from_numpy(image).type(dtype=torch.float)

parser = argparse.ArgumentParser()
parser.add_argument('--trainA_path',type=str,default='./data/trainA')
parser.add_argument('--trainB_path',type=str,default='./data/trainB')
parser.add_argument('--use_wgan',type=bool,default=True,help='Use WGAN to train')
parser.add_argument('--lr',type=float,default=1e-4,help='learning rate')
parser.add_argument('--max_epoch',type=int,default=500,help='Max epoch for training')
parser.add_argument('--bz',type=int,default=32,help='batch size for training')
parser.add_argument('--lbda1',type=int,default=100,help='weight for L1 loss')
parser.add_argument('--lbda2',type=int,default=1,help='weight for iamge gradient difference loss')
parser.add_argument('--num_workers',type=int,default=4,help='Use multiple kernels to load dataset')
parser.add_argument('--checkpoints_root',type=str,default='checkpoints',help='The root path to save checkpoints')
parser.add_argument('--log_root',type=str,default='./log',help='The root path to save log files which are writtern by tensorboardX')
parser.add_argument('--gpu_id',type=str,default='0',help='Choose one gpu to use. Only single gpu training is supported currently')
args = parser.parse_args()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    wgan = args.use_wgan
    learnint_rate = args.lr
    max_epoch = args.max_epoch
    batch_size = args.bz
    lambda_1 = args.lbda1
    lambda_2 = args.lbda2 # Weight for image gradient difference loss

    netG = GeneratorNet().cuda()
    netD = DiscrimiterNet(wgan_loss=wgan).cuda()

    optimizer_g = optim.Adam(netG.parameters(),lr=learnint_rate)
    optimizer_d = optim.Adam(netD.parameters(),lr=learnint_rate)

    dataset = MYDataSet(src_data_path=args.trainA_path,dst_data_path=args.trainB_path)
    datasetloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=args.num_workers)

    log_root = args.log_root
    date = datetime.datetime.now().strftime('%F_%T').replace(':','_')
    log_folder = date
    log_dir = os.path.join(log_root,log_folder)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    checkpoint_root = args.checkpoints_root
    checkpoint_folder = date
    checkpoint_dir = os.path.join(checkpoint_root,checkpoint_folder)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(0,max_epoch):
        d_loss_log_list = []
        g_loss_log_list = []
        for iteration, data in enumerate(datasetloader):
            batchtensor_A = data[0].cuda()
            batchtensor_B = data[1].cuda()
            generated_batchtensor = netG.forward(batchtensor_A)

            ######################
            # (1) Train Discriminator
            ######################
            num_critic = 1
            if wgan:
                num_critic = 5
            for i in range(num_critic):
                optimizer_d.zero_grad()
                d_fake = netD(generated_batchtensor)
                d_real = netD(batchtensor_B)

                #------------------------------#    
                #--- wgan loss cost function---#
                d_loss = torch.mean(d_fake) - torch.mean(d_real) # E[D(I_C)] = E[D(G(I_D))]
                
                lambda_gp = 10 # as setted in the paper
                
                epsilon = torch.rand(batchtensor_B.size()[0], 1, 1, 1).cuda()
                x_hat = batchtensor_B * epsilon + (1 -epsilon)*generated_batchtensor
                d_hat = netD.forward(x_hat)
                
                # Following code is taken from https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
                # to calculate gradients penalty
                grad_outputs = torch.ones(d_hat.size()).cuda()
                gradients = torch.autograd.grad( # Calculate gradients of probabilities with respect to examples
                    outputs=d_hat,
                    inputs=x_hat,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True
                )[0]
                # Gradients have shape (batch_size, num_channels, img_width, img_height),
                # so flatten to easily take norm per example in batch
                gradients = gradients.view(batch_size,-1)
                
                # Derivatives of the gradient close to 0 can cause problems because of
                # the square root, so manually calculate norm and add epsilon
                gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

                # Calculate gradient penalty
                gradient_penalty =  lambda_gp*torch.mean((gradients_norm - 1) ** 2)

                d_loss += gradient_penalty 
                #--- wgan loss cost function---#
                #------------------------------#   

                d_loss.backward(retain_graph=True)
                netG.zero_grad()
                optimizer_d.step()
                d_loss_log = d_loss.item()
                d_loss_log_list.append(d_loss_log)

            ######################
            # (2) Train G network
            ######################
            optimizer_g.zero_grad()
            d_fake = netD(generated_batchtensor)
            
            g_loss = -torch.mean(d_fake)
            base_loss_log = g_loss.item()
            l1_loss =  torch.mean(torch.abs(generated_batchtensor-batchtensor_B))
            l1_loss_log = l1_loss.item()
            igdl_loss = loss_igdl(batchtensor_B,generated_batchtensor)
            igdl_loss_log = igdl_loss.item()

            g_loss += lambda_1 *l1_loss + lambda_2*igdl_loss
            g_loss += lambda_1 * l1_loss
            g_loss_log = g_loss.item()
            g_loss_log_list.append(g_loss_log)

            g_loss.backward()
            netD.zero_grad()
            optimizer_g.step()

            writer.add_scalar('G_loss',g_loss_log,(epoch*len(datasetloader)+iteration))
            writer.add_scalar('D_loss',d_loss_log,(epoch*len(datasetloader)+iteration))
            writer.add_scalar('base_loss',base_loss_log,(epoch*len(datasetloader)+iteration))
            writer.add_scalar('l1_loss',l1_loss_log,(epoch*len(datasetloader)+iteration))
            writer.add_scalar('IGDL_loss',igdl_loss_log,(epoch*len(datasetloader)+iteration))
            print('==>Epoch:%d/%d:%d d_loss:%.3f g_loss:%.3f'%(epoch,max_epoch,iteration,d_loss_log,g_loss_log))

        d_loss_average_log = np.array(d_loss_log_list).mean()
        g_loss_average_log = np.array(g_loss_log_list).mean()

        writer.add_scalar('D_loss_epoch',d_loss_average_log,epoch)
        writer.add_scalar('G_loss_epoch',g_loss_average_log,epoch)

        torch.save(netD.state_dict(),os.path.join(checkpoint_dir,'netD_%d.pth'%epoch))
        torch.save(netG.state_dict(),os.path.join(checkpoint_dir,'netG_%d.pth'%epoch))
    
    writer.close()
    

