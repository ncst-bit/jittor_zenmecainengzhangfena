import torch
import jittor as jt

def convert_model(input_path,output_path,mode):

    if mode=='0':
        weights = torch.load(input_path,map_location=torch.device('cpu')).state_dict()
    if mode=='1':
        weights = torch.load(input_path, map_location=torch.device('cpu'))
    for k in weights.keys():
        weights[k] = weights[k].float().cpu()
    jt.save(weights, output_path)

#https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt
convert_model('./publicModel/RN101.pt','./publicModel/RN101.pkl','0')

#https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
convert_model('./publicModel/ViT-B-32.pt','./publicModel/ViT-B-32.pkl','0')

#https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth
convert_model('./publicModel/efficientnet-b6-c76e70fd.pth','./publicModel/efficientnet-b6-c76e70fd.pkl','1')
