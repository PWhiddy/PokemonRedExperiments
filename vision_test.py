print('importing libs...')

import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw
import requests
from io import BytesIO

print('loading model...')
res34 = torchvision.models.resnet34(pretrained=True, progress=True, num_classes=1000)
res34.eval()

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print('loading image...')
input_images = torchvision.datasets.ImageFolder(
        root='./test_images/',
        transform=preprocess
    )


model_input = input_images[1][0].unsqueeze(0) #.reshape(1,3,1600,1200)

#print(input_images[0][0].reshape(1,3,1600,1200).size())

url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/' \
      'raw/596b27d23537e5a1b5751d2b0481ef172f58b539/imagenet1000_clsid_to_human.txt'

'''
images_url = [('cat', 'https://www.wired.com/wp-content/uploads/2015/02/catinbox_cally_by-helen-haden_4x31-660x495.jpg'),
    ('pomeranian', 'https://c.photoshelter.com/img-get/I0000q_DdkyvP6Xo/s/900/900/Pomeranian-Dog-with-Ball.jpg'),
    ('boat', 'http://gasequipment.com.au/wp-content/uploads/2016/03/Boat-for-Blogpost.jpg')]

response = requests.get(images_url[1][1])
im = Image.open(BytesIO(response.content))
tens = preprocess(im)
model_input = tens.unsqueeze(0)
'''

print('reading classes...')
imagenet_classes = eval(requests.get(url).content)

print('model inference...')
raw_result = res34(model_input)

probs = torch.nn.functional.softmax(raw_result[0], dim=0)
res_index = probs.argmax().item()
print('index: ', res_index)
print('res50 predicted ', imagenet_classes[res_index], ' confidence: ', probs[res_index].item(), '\n')