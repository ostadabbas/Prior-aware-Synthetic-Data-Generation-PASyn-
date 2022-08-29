import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import models.transformer as transformer
import models.StyTR as StyTR
import matplotlib.pyplot as plt
from matplotlib import cm
from function import normal
import numpy as np
import time
from collections import OrderedDict
import sys

np.set_printoptions(threshold=sys.maxsize)

class StyTR_API:
    def __init__(self):
        self.vgg_pth = './experiments/vgg_normalised.pth'
        self.decoder_pth = 'experiments/decoder_iter_160000.pth'
        self.trans_pth = 'experiments/transformer_iter_160000.pth'
        self.emb_pth = 'experiments/embedding_iter_160000.pth'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vgg = StyTR.vgg
        self.vgg.load_state_dict(torch.load(self.vgg_pth))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:44])
        self.decoder = StyTR.decoder
        self.Trans = transformer.Transformer()
        self.embedding = StyTR.PatchEmbed()
        self.decoder.eval()
        self.Trans.eval()
        self.vgg.eval()
        new_state_dict = OrderedDict()
        state_dict = torch.load(self.decoder_pth)
        for k, v in state_dict.items():
            namekey = k
            new_state_dict[namekey] = v
        self.decoder.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        state_dict = torch.load(self.trans_pth)
        for k, v in state_dict.items():
            namekey = k
            new_state_dict[namekey] = v
        self.Trans.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        state_dict = torch.load(self.emb_pth)
        for k, v in state_dict.items():
            namekey = k
            new_state_dict[namekey] = v
        self.embedding.load_state_dict(new_state_dict)

        network = StyTR.StyTrans(self.vgg,self.decoder,self.embedding,self.Trans,None)
        network.eval()
        network.to(self.device)
        self.network = network
        self.content_tf = self.test_transform(512, 'store_true')
        self.style_tf = self.test_transform(512, 'store_true')
        self.output2_PIL_T = transforms.ToPILImage()


    def gen_image(self, content_pth, style_pth, output_path, alpha):
        save_ext = ".png"
        content_paths = [Path(content_pth)]
        style_paths = [Path(style_pth)]
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for content_path in content_paths:
            for style_path in style_paths:
                content_tf1 = self.content_transform()       
                content = self.content_tf(Image.open(content_path).convert("RGB"))

                h,w,c=np.shape(content)    
                style_tf1 = self.style_transform(h,w)
                style = self.style_tf(Image.open(style_path).convert("RGB"))
            
                style = style.to(self.device).unsqueeze(0)
                content = content.to(self.device).unsqueeze(0)
                
                with torch.no_grad():
                    output = self.network(content,style)       
                output = output[0].cpu()
                output2 = content.cpu() * alpha + output * (1-alpha)
                # output2 = (torch.floor(output2 * 100) / 100)
                #output2 = ((torch.floor(output2 * 100) / 100) * 255 - 20) / 255
                #output2 = output2 * 255
                #output2 = output2.int()
                #output2[output2<0] = 0
                #output2[output2>255] = 255
                #output2_PIL = self.output2_PIL_T(output2[0])
                output2 = output2[0].numpy().transpose(1,2,0)
                output2[:,:,0] = (output2[:,:,0]*255/np.max(output2[:,:,0]))-1
                output2[:,:,1] = (output2[:,:,1]*255/np.max(output2[:,:,1]))-1
                output2[:,:,2] = (output2[:,:,2]*255/np.max(output2[:,:,2]))-1
                output2[output2<0] = 0
                print(np.max(output2[:,:,0]), np.max(output2[:,:,1]), np.max(output2[:,:,2]))
                output2 = output2.astype(np.uint8)
                #a = np.array(output2)
                output2_PIL = Image.fromarray(output2)
                
                
                # Image.fromarray(output.numpy()[0]).save("duc.jpg")
                        
                #output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                #    output_path, splitext(basename(content_path))[0],
                #    splitext(basename(style_path))[0], save_ext
                #)
                #output2_name = '{:s}/{:s}_stylized_new_{:s}{:s}'.format(
                #    output_path, splitext(basename(content_path))[0],
                #    splitext(basename(style_path))[0], save_ext
                #)
        
                #save_image(output, output_name)
                content = Image.open(content_path)
                var = Image.Image.split(content)
                var2 = Image.Image.split(output2_PIL.resize((300,300)))
                res = Image.merge('RGBA',(var2[0],var2[1],var2[2],var[3]))
                return res
        
    def test_transform(self, size, crop):
        transform_list = []
    
        if size != 0: 
            transform_list.append(transforms.Resize(size))
        if crop:
            transform_list.append(transforms.CenterCrop(size))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform

    def style_transform(self,h,w):
        k = (h,w)
        size = int(np.max(k))
        print(type(size))
        transform_list = []    
        transform_list.append(transforms.CenterCrop((h,w)))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform

    def content_transform(self):
        transform_list = []   
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform

if __name__ == '__main__':
    api = StyTR_API()
    #img = api.gen_image("../dataset/data/0000.png", "../dataset/background/2.jpg", "out")


