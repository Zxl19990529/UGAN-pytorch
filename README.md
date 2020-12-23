# Underwater GAN (UGAN) pytorch

[[Original tensorflow version](https://github.com/cameronfabbri/Underwater-Color-Correction)] [[Project page](http://irvlab.cs.umn.edu/enhancing-underwater-imagery-using-gans)]

This is a pytorch implementation of [Enhancing Underwater Imagery using Generative Adversarial Networks](https://arxiv.org/pdf/1801.04011.pdf). In this repo, we only implement the UGAN-GP.

## Usage

- Environment:
    - python3.7
    - pytorch1.6
    - tensorboardX
    - opencv-python
    - cuda
    - anaconda3
The environment can be install by commanding ```conda env create -f pytorch16.yaml```

### Dataset preparing

UGAN is an end to end network, it aims at learning a map from imageA to imageB. 1) Download [Underwater Imagenet](https://drive.google.com/file/d/1LOM-2A1BSLaFjCY2EEK3DA2Lo37rNw-7/view) 2) Unzip it to data folder. Then the data folder should be organized as :
```py
data
    - test
    - trainA
    - trainB
```
### Training

All args:
```py
parser = argparse.ArgumentParser()
parser.add_argument('--trainA_path',type=str,default='./data/trainA')
parser.add_argument('--trainB_path',type=str,default='./data/trainB')
parser.add_argument('--use_wgan',type=bool,default=True,help='Use WGAN to train')
parser.add_argument('--lr',type=float,default=1e-4,help='learning rate')
parser.add_argument('--max_epoch',type=int,default=300,help='Max epoch for training')
parser.add_argument('--bz',type=int,default=32,help='batch size for training')
parser.add_argument('--lbda1',type=int,default=100,help='weight for L1 loss')
parser.add_argument('--lbda2',type=int,default=1,help='weight for iamge gradient difference loss')
parser.add_argument('--num_workers',type=int,default=4,help='Use multiple kernels to load dataset')
parser.add_argument('--checkpoints_root',type=str,default='checkpoints',help='The root path to save checkpoints')
parser.add_argument('--log_root',type=str,default='./log',help='The root path to save log files which are writtern by tensorboardX')
parser.add_argument('--gpu_id',type=str,default='0',help='Choose one gpu to use. Only single gpu training is supported currently')

```

Example:```python train.py --trainA_path ./data/trainA --trainB_path ./data/trainB --use_wgan True --lr 1e-4 --max_epoch 500 --bz 32 --lbda1 100 --lbad2 1 --num_workers 4 --checkpoints_root ./checkpoints --log_root ./log --gpu_id 0```

To trace the training progress, use tensorboardX by commanding ```tensorboard --logdir log/year-month-date_hour_minute_second```.

### Eval

To evaluate one image:
```python eval_one.py --img_path ***.jpg --checkpoint ./checkpoints/netG_**.pth```
To evaluate images in folder:
```python eval_folder --img_folder ./data/test --checkpoint ./checkpoints/netG_**.pth --output_folder ./output```

Note: Due to the network archeticture, the input image's height and width should be integer multiples of 256.

### Result

In batchsize 32, I trained 300 epochs. However, it can continue be trained for better results.

![](./readme_imgs/original_img.png)
![](./readme_imgs/tensorlfow_version.png)
![](./readme_imgs/my_result.png)

