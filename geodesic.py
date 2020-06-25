import torch
from torch import nn
import numpy as np
from torch import optim
import sys
from generator import Generator

geo_D = np.zeros((400,400))
Z = np.load('./Z.npy')#Z is randomly sampled points in latent space
Z = torch.Tensor(Z)
    
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
    #start from linear interpolation
    z_mid = z_lin[1:-1].clone().to(device)#points except 2 ends are changeable
    z_s = z_lin[0].clone().unsqueeze(0).to(device)#starting point
    z_e = z_lin[-1].clone().unsqueeze(0).to(device)#ending point
    z_mid.requires_grad = True
    opt = optim.SGD([z_mid],lr=lr,momentum=0.9)
    
    iteration = 5000
    for it in range(iteration):
        #map to image space
        G_s = G(z_s)
        G_e = G(z_e)
        G_mid = G(z_mid)
        #energy
        E = torch.sum((G_s-G_mid[0])**2)+torch.sum((G_mid[:-1]-G_mid[1:])**2)+torch.sum((G_mid[-1]-G_e)**2)
        E.backward()
        opt.step()
        if it%10==0:
            print(it,float(torch.norm(z_mid.grad.data)),float(E),flush=True)
        #stop when gradinets are relatively small
        if torch.norm(z_mid.grad.data)<10:
            break
        z_mid.grad.data.zero_()
    return torch.cat([z_s,z_mid,z_e],0)

def main():
    for i in range(399):
        for j in range(i+1,400):
            
            print(i,j,flush=True)
            z1 = Z[i].unsqueeze(0)
            z2 = Z[j].unsqueeze(0)
            z_lin = []
            #linear interpolation
            for k in range(11):
                zt = z1*0.1*k+z2*0.1*(10-k)
                z_lin.append(zt)
            z_lin = torch.cat(z_lin,0)
        
            z_geo = get_geo(z_lin,1e-4)
            #map geodesic to img space
            geo_imgs = get_imgs(z_geo.unsqueeze(1))
            geo_len = get_arc_len(geo_imgs)
        
            geo_D[i][j] = geo_len
            geo_D[j][i] = geo_len
        
            np.save('./geo_D.npy',geo_D)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G = Generator(ngpu=1).eval().to(device)
G.load_state_dict(torch.load('./netG_epoch_199.pth'))#load trained generator
for p in G.parameters():
    print(float(torch.norm(p)))
main()
