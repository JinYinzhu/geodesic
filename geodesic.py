import torch
from torch import nn
import numpy as np
from torch import optim
import sys

#lin_D = np.zeros((400,400))
#geo_D = np.zeros((400,400))
lin_D = np.load('./geo_D.npy')
geo_D = np.load('./lin_D.npy')
Z = np.load('./Z.npy')
Z = torch.Tensor(Z)

class Generator(nn.Module):
    def __init__(self,ngpu,nc=3,nz=100,ngf=64):
        super(Generator,self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz,ngf*8,4,1,0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            #(ngf*8)x4x4
            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            #(ngf*4)x8x8
            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            #(ngf*2)x16x16
            nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf,nc,kernel_size=1,stride=1,padding=0,bias=False),
            nn.Tanh()
        )

    def forward(self,input):
        if input.is_cuda and self.ngpu>1:
            output = nn.parallel.data_parallel(self.main,input,range(self.ngpu))
        else:
            output = self.main(input)
        return output
    
def get_arc_len(imgs):
    vc_lst = [img.view(-1) for img in imgs]
    leng = 0
    for i in range(1,len(vc_lst)):
        vc1 = vc_lst[i]
        vc2 = vc_lst[i-1]
        leng += torch.norm(vc1-vc2)
    return leng

def get_imgs(z_geo):
    img_lst = []
    for zi in z_geo:
        xt = G.forward(zi)
        xt = xt.detach()
        img_lst.append(xt)
    imgs_geo = torch.cat(img_lst,0)
    return imgs_geo

def get_geo(z_lin,lr):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z_mid = z_lin[1:-1].clone().to(device)
    z_s = z_lin[0].clone().unsqueeze(0).to(device)
    z_e = z_lin[-1].clone().unsqueeze(0).to(device)
    z_mid.requires_grad = True
    opt = optim.SGD([z_mid],lr=lr,momentum=0.9)
    iteration = 5000
    for it in range(iteration):
        G_s = G.forward(z_s)
        G_e = G.forward(z_e)
        G_mid = G.forward(z_mid)
        E = torch.sum((G_s-G_mid[0])**2)+torch.sum((G_mid[:-1]-G_mid[1:])**2)+torch.sum((G_mid[-1]-G_e)**2)
        E.backward()
        opt.step()
        if it%10==0:
            print(it,float(torch.norm(z_mid.grad.data)),float(E),flush=True)
        if torch.norm(z_mid.grad.data)<10:
            break
        z_mid.grad.data.zero_()
    return torch.cat([z_s,z_mid,z_e],0)

def main():
    for i in range(92,399):
        for j in range(i+1,400):
            if i==92 and j<298:
                continue
            
            print(i,j,flush=True)
            z1 = Z[i].unsqueeze(0)
            z2 = Z[j].unsqueeze(0)
            z_lin = []
            for k in range(11):
                zt = z1*0.1*k+z2*0.1*(10-k)
                z_lin.append(zt)
            z_lin = torch.cat(z_lin,0)
        
            z_geo = get_geo(z_lin,1e-4)
            geo_imgs = get_imgs(z_geo.unsqueeze(1))
            geo_len = get_arc_len(geo_imgs)
            lin_dst = torch.norm(geo_imgs[0]-geo_imgs[-1])
            print(lin_dst)
            for p in G.parameters():
                print(float(torch.norm(p)))
            #return
        
            #geo_D[i][j] = geo_len
            #geo_D[j][i] = geo_len
            #lin_D[i][j] = lin_dst
            #lin_D[j][i] = lin_dst
        
            #np.save('./lin_D.npy',lin_D)
            #np.save('./geo_D.npy',geo_D)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('started',flush=True)
G = Generator(ngpu=1).eval().to(device)
G.load_state_dict(torch.load('./netG_epoch_199.pth',map_location=torch.device('cpu')))
for p in G.parameters():
    print(float(torch.norm(p)))
main()
