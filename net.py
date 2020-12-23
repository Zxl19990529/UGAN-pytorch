import torch
import torch.nn as nn 
from torch.nn import Conv2d,LeakyReLU,BatchNorm2d, ConvTranspose2d,ReLU
# from utils import normal_init

def encoder_layer(in_channels,out_channels,kernel_size=4,stride = 2,padding = 1): # NOTE: Padding here is different from the 'vaild' in tensorflow version of original github
    layer = nn.Sequential(
        Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
        BatchNorm2d(out_channels),
        LeakyReLU(0.2)
    )
    return layer

def decoder_layer(in_channels,out_channels,last_layer=False,kernel_size=4,stride = 2,padding = 1):
    if not last_layer:
        layer = nn.Sequential(
            ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
            torch.nn.ReLU()
        )
    else:
        layer = nn.Sequential(
            ConvTranspose2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1),
            torch.nn.Tanh()
        )
    return layer

def discrimiter_layer(in_channels,out_channels,kernel_size=4,stride = 2,padding = 1,wgan=False):
    if wgan:
        layer = nn.Sequential(
            Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
            BatchNorm2d(out_channels),
            LeakyReLU(0.2)
        )
    else:
        layer = nn.Sequential(
            Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
            LeakyReLU(0.2)
        )
    return layer


class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()

        # Encoder
        self.enc_conv1 = encoder_layer(3,64)
        self.enc_conv2 = encoder_layer(64,128)
        self.enc_conv3 = encoder_layer(128,256)
        self.enc_conv4 = encoder_layer(256,512)
        self.enc_conv5 = encoder_layer(512,512)
        self.enc_conv6 = encoder_layer(512,512)
        self.enc_conv7 = encoder_layer(512,512)
        self.enc_conv8 = encoder_layer(512,512,padding=1)
        # Decoder
        self.dec_conv1 = decoder_layer(512,512)
        self.dec_conv2 = decoder_layer(1024,512)
        self.dec_conv3 = decoder_layer(1024,512)
        self.dec_conv4 = decoder_layer(1024,512)
        self.dec_conv5 = decoder_layer(1024,256)
        self.dec_conv6 = decoder_layer(512,128)
        self.dec_conv7 = decoder_layer(256,64)
        self.dec_conv8 = decoder_layer(128,3,last_layer=True)
        
    def forward(self,input_x):
        # Encoder
        output_enc_conv1 = self.enc_conv1(input_x)
        output_enc_conv2 = self.enc_conv2(output_enc_conv1)
        output_enc_conv3 = self.enc_conv3(output_enc_conv2)
        output_enc_conv4 = self.enc_conv4(output_enc_conv3)
        output_enc_conv5 = self.enc_conv5(output_enc_conv4)
        output_enc_conv6 = self.enc_conv6(output_enc_conv5)
        output_enc_conv7 = self.enc_conv7(output_enc_conv6)
        output_enc_conv8 = self.enc_conv8(output_enc_conv7)


        #  Decoder
        output_dec_conv1 = self.dec_conv1(output_enc_conv8)
        output_dec_conv1 = torch.cat([output_dec_conv1,output_enc_conv7],dim = 1)

        output_dec_conv2 = self.dec_conv2(output_dec_conv1)
        output_dec_conv2 = torch.cat([output_dec_conv2,output_enc_conv6],dim = 1)

        output_dec_conv3 = self.dec_conv3(output_dec_conv2)
        output_dec_conv3 = torch.cat([output_dec_conv3,output_enc_conv5],dim = 1)

        output_dec_conv4 = self.dec_conv4(output_dec_conv3)
        output_dec_conv4 = torch.cat([output_dec_conv4,output_enc_conv4],dim = 1)

        output_dec_conv5 = self.dec_conv5(output_dec_conv4)
        output_dec_conv5 = torch.cat([output_dec_conv5,output_enc_conv3],dim = 1)

        output_dec_conv6 = self.dec_conv6(output_dec_conv5)
        output_dec_conv6 = torch.cat([output_dec_conv6,output_enc_conv2],dim = 1)

        output_dec_conv7 = self.dec_conv7(output_dec_conv6)
        output_dec_conv7 = torch.cat([output_dec_conv7,output_enc_conv1],dim = 1)

        output_dec_conv8 = self.dec_conv8(output_dec_conv7)

        return output_dec_conv8
      
    
class DiscrimiterNet(torch.nn.Module):
    def __init__(self,wgan_loss):
        super(DiscrimiterNet, self).__init__()
        self.wgan_loss = wgan_loss

        self.conv1 = discrimiter_layer(3,64,self.wgan_loss)
        self.conv2 = discrimiter_layer(64,128,self.wgan_loss)
        self.conv3 = discrimiter_layer(128,256,self.wgan_loss)
        self.conv4 = discrimiter_layer(256,512,self.wgan_loss)
        self.conv5 = discrimiter_layer(512,1,kernel_size=1,stride=1)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x