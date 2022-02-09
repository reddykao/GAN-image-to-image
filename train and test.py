# ab_004.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torchvision.utils as vutils
from micro_lab_model import Generator, Discriminator
from PIL import Image
import torchvision

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#device= torch.device('cuda' )
print('Using device:', device)
print()


'''
if torch.cuda.is_available():
    torch.device('cuda')
    print('gpu')
else:
    torch.device('cpu')
    print('cpu')
'''   
class MyDataset(Dataset):
    def __init__(self, root, subfolder, transform=None):
        super(MyDataset, self).__init__()
        self.path = os.path.join(root, subfolder)
        self.image_list = [x for x in os.listdir(self.path)]
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image_path = os.path.join(self.path, self.image_list[item])
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  #BGR->RGB
        if self.transform is not None:
            image = self.transform(image)
            
        lable = 'NONE'
        return image, lable

#-------------------------------------------------------
def loadData(root, subfolder, batch_size, shuffle=True):
    # 準備訓練數據
    transform = transforms.Compose([
        transforms.ToTensor(),  # (H, W, C) -> (C, H, W) & (0, 255) -> (0, 1)
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # (0, 1) -> (-1, 1)
    ])
    dataset = MyDataset(root, subfolder, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def split(img): 
    return img[:,:,:,:256], img[:,:,:,256:] 

#---------------------------------------
root = 'H:/SSIM/microlab/image/0125mr/AB/'
subfolder = 'train'
batch_size = 10
train_loader = loadData(root, subfolder, batch_size, shuffle=False)
#for dx, _ in train_loader:
r_train, _ = next(iter(train_loader))
#y, X = split(r_train)

#----------------------------------------------
# Models
G = Generator(3, 64, 3)
D = Discriminator(6, 64, 1)
G.normal_weight_init(mean=0.0, std=0.02)
D.normal_weight_init(mean=0.0, std=0.02)

# Loss function
BCE_loss = torch.nn.BCELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()

# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.99))
D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.99))

# Training GAN
D_avg_losses = []
G_avg_losses = []

step = 0
num_epochs = 200

for epoch in range(num_epochs):
    D_losses = []
    G_losses = []
    
    # training
    for dx_, _ in train_loader:
        step += 1
        x_, y_ = split(dx_)
        print(step)
        # Train discriminator with real data
        D_real_decision = D(x_, y_).squeeze()
        real_ = Variable(torch.ones(D_real_decision.size()))
        D_real_loss = BCE_loss(D_real_decision, real_)

        # Train discriminator with fake data
        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        fake_ = Variable(torch.zeros(D_fake_decision.size()))
        D_fake_loss = BCE_loss(D_fake_decision, fake_)

        # Back propagation
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        G_fake_loss = BCE_loss(D_fake_decision, real_)

        # L1 loss
        l1_loss = 100 * L1_loss(gen_image, y_)

        # Back propagation
        G_loss = G_fake_loss + l1_loss
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

#----------------------------------------------
def show_E2S(batch1, batch2, title1, title2):
    # edges
    plt.figure(figsize=(7,7))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title(title1)
    plt.imshow(np.transpose(vutils.make_grid(batch1, nrow=1, padding=5, 
    normalize=True).cpu(),(1,2,0)))
    # shoes
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title(title2)
    plt.imshow(np.transpose(vutils.make_grid(batch2, nrow=1, padding=5, 
    normalize=True).cpu(),(1,2,0)))
    
    #plt.savefig('plot.png')
    #plt.show()

show_E2S(gen_image,y_,"predict X ","ground truth")


torchvision.utils.save_image(gen_image[0],'H:/SSIM/microlab/result/0208/GENERATE/0.png',normalize=True)
torchvision.utils.save_image(y_[0],'H:/SSIM/microlab/result/0208/GT/0.png',normalize=True)

torchvision.utils.save_image(gen_image[1],'H:/SSIM/microlab/result/0208/GENERATE/1.png',normalize=True)
torchvision.utils.save_image(y_[1],'H:/SSIM/microlab/result/0208/GT/1.png',normalize=True)

torchvision.utils.save_image(gen_image[2],'H:/SSIM/microlab/result/0208/GENERATE/2.png',normalize=True)
torchvision.utils.save_image(y_[2],'H:/SSIM/microlab/result/0208/GT/2.png',normalize=True)

torchvision.utils.save_image(gen_image[3],'H:/SSIM/microlab/result/0208/GENERATE/3.png',normalize=True)
torchvision.utils.save_image(y_[3],'H:/SSIM/microlab/result/0208/GT/3.png',normalize=True)

torchvision.utils.save_image(gen_image[4],'H:/SSIM/microlab/result/0208/GENERATE/4.png',normalize=True)
torchvision.utils.save_image(y_[4],'H:/SSIM/microlab/result/0208/GT/4.png',normalize=True)

torchvision.utils.save_image(gen_image[5],'H:/SSIM/microlab/result/0208/GENERATE/5.png',normalize=True)
torchvision.utils.save_image(y_[5],'H:/SSIM/microlab/result/0208/GT/5.png',normalize=True)

torchvision.utils.save_image(gen_image[6],'H:/SSIM/microlab/result/0208/GENERATE/6.png',normalize=True)
torchvision.utils.save_image(y_[6],'H:/SSIM/microlab/result/0208/GT/6.png',normalize=True)

torchvision.utils.save_image(gen_image[7],'H:/SSIM/microlab/result/0208/GENERATE/7.png',normalize=True)
torchvision.utils.save_image(y_[7],'H:/SSIM/microlab/result/0208/GT/7.png',normalize=True)

torchvision.utils.save_image(gen_image[8],'H:/SSIM/microlab/result/0208/GENERATE/8.png',normalize=True)
torchvision.utils.save_image(y_[8],'H:/SSIM/microlab/result/0208/GT/8.png',normalize=True)

torchvision.utils.save_image(gen_image[9],'H:/SSIM/microlab/result/0208/GENERATE/9.png',normalize=True)
torchvision.utils.save_image(y_[9],'H:/SSIM/microlab/result/0208/GT/9.png',normalize=True)
'''

torchvision.utils.save_image(gen_image[10],'H:/SSIM/microlab/result/0208/GENERATE/10.png',normalize=True)
torchvision.utils.save_image(y_[10],'H:/SSIM/microlab/result/0208/GT/10.png',normalize=True)

torchvision.utils.save_image(gen_image[11],'H:/SSIM/microlab/result/0208/GENERATE/11.png',normalize=True)
torchvision.utils.save_image(y_[11],'H:/SSIM/microlab/result/0208/GT/11.png',normalize=True)

torchvision.utils.save_image(gen_image[12],'H:/SSIM/microlab/result/0208/GENERATE/12.png',normalize=True)
torchvision.utils.save_image(y_[12],'H:/SSIM/microlab/result/0208/GT/12.png',normalize=True)

torchvision.utils.save_image(gen_image[13],'H:/SSIM/microlab/result/0208/GENERATE/13.png',normalize=True)
torchvision.utils.save_image(y_[13],'H:/SSIM/microlab/result/0208/GT/13.png',normalize=True)

torchvision.utils.save_image(gen_image[14],'H:/SSIM/microlab/result/0208/GENERATE/14.png',normalize=True)
torchvision.utils.save_image(y_[14],'H:/SSIM/microlab/result/0208/GT/14.png',normalize=True)

torchvision.utils.save_image(gen_image[15],'H:/SSIM/microlab/result/0208/GENERATE/15.png',normalize=True)
torchvision.utils.save_image(y_[15],'H:/SSIM/microlab/result/0208/GT/15.png',normalize=True)

torchvision.utils.save_image(gen_image[16],'H:/SSIM/microlab/result/0208/GENERATE/16.png',normalize=True)
torchvision.utils.save_image(y_[16],'H:/SSIM/microlab/result/0208/GT/16.png',normalize=True)

torchvision.utils.save_image(gen_image[17],'H:/SSIM/microlab/result/0208/GENERATE/17.png',normalize=True)
torchvision.utils.save_image(y_[17],'H:/SSIM/microlab/result/0208/GT/17.png',normalize=True)

torchvision.utils.save_image(gen_image[18],'H:/SSIM/microlab/result/0208/GENERATE/18.png',normalize=True)
torchvision.utils.save_image(y_[18],'H:/SSIM/microlab/result/0208/GT/18.png',normalize=True)

torchvision.utils.save_image(gen_image[19],'H:/SSIM/microlab/result/0208/GENERATE/19.png',normalize=True)
torchvision.utils.save_image(y_[19],'H:/SSIM/microlab/result/0208/GT/19.png',normalize=True)


torchvision.utils.save_image(gen_image[20],'H:/SSIM/microlab/result/0208/GENERATE/20.png',normalize=True)
torchvision.utils.save_image(y_[20],'H:/SSIM/microlab/result/0208/GT/20.png',normalize=True)

torchvision.utils.save_image(gen_image[21],'H:/SSIM/microlab/result/0208/GENERATE/21.png',normalize=True)
torchvision.utils.save_image(y_[21],'H:/SSIM/microlab/result/0208/GT/21.png',normalize=True)

torchvision.utils.save_image(gen_image[22],'H:/SSIM/microlab/result/0208/GENERATE/22.png',normalize=True)
torchvision.utils.save_image(y_[22],'H:/SSIM/microlab/result/0208/GT/22.png',normalize=True)

torchvision.utils.save_image(gen_image[23],'H:/SSIM/microlab/result/0208/GENERATE/23.png',normalize=True)
torchvision.utils.save_image(y_[23],'H:/SSIM/microlab/result/0208/GT/23.png',normalize=True)

torchvision.utils.save_image(gen_image[24],'H:/SSIM/microlab/result/0208/GENERATE/24.png',normalize=True)
torchvision.utils.save_image(y_[24],'H:/SSIM/microlab/result/0208/GT/24.png',normalize=True)

torchvision.utils.save_image(gen_image[25],'H:/SSIM/microlab/result/0208/GENERATE/25.png',normalize=True)
torchvision.utils.save_image(y_[25],'H:/SSIM/microlab/result/0208/GT/25.png',normalize=True)

torchvision.utils.save_image(gen_image[26],'H:/SSIM/microlab/result/0208/GENERATE/26.png',normalize=True)
torchvision.utils.save_image(y_[26],'H:/SSIM/microlab/result/0208/GT/26.png',normalize=True)

torchvision.utils.save_image(gen_image[27],'H:/SSIM/microlab/result/0208/GENERATE/27.png',normalize=True)
torchvision.utils.save_image(y_[27],'H:/SSIM/microlab/result/0208/GT/27.png',normalize=True)

torchvision.utils.save_image(gen_image[28],'H:/SSIM/microlab/result/0208/GENERATE/28.png',normalize=True)
torchvision.utils.save_image(y_[28],'H:/SSIM/microlab/result/0208/GT/28.png',normalize=True)

torchvision.utils.save_image(gen_image[29],'H:/SSIM/microlab/result/0208/GENERATE/29.png',normalize=True)
torchvision.utils.save_image(y_[29],'H:/SSIM/microlab/result/0208/GT/29.png',normalize=True)

'''

