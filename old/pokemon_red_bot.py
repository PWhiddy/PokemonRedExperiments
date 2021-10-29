import sys
from pyboy import PyBoy, WindowEvent
import torch
import torchvision
import hnswlib
import numpy as np

vec_dim = 1080 #4320 #1000
num_elements = 5000 # max
all_stored_frame_vecs = []

# Generating sample data
#data = np.float32(np.random.random((num_elements, dim)))
#data_labels = np.arange(num_elements)

# Declaring index
p = hnswlib.Index(space = 'l2', dim = vec_dim) # possible options are l2, cosine or ip

# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 100, M = 16)

# Controlling the recall by setting ef:
#p.set_ef(50) # ef should always be > k

pyboy = PyBoy(
        "./PokemonRed.gb",
        debugging=False,
        disable_input=False,
        # window_type="headless", # For unattended use, for example machine learning
        window_type='SDL2',
        hide_window="--quiet" in sys.argv,
    )


with open("init.state", "rb") as f:
    pyboy.load_state(f)

pyboy.set_emulation_speed(0)

preprocess = torchvision.transforms.Compose([
#     torchvision.transforms.Resize(224),
#    torchvision.transforms.CenterCrop(224),
#    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

res34 = torchvision.models.resnet34(pretrained=True, progress=True, num_classes=1000)
res34.eval()

frame = 0
while not pyboy.tick():

    '''
    if frame%200 == 0:
        print('pressing start...')
        pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
    else:
        pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

    if frame%5 == 0:
        print('pressing A...')
        pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
    else:
        pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
    '''

    if frame % 5 == 0:
        #print(pyboy.get_raw_screen_buffer())
        pix = torch.from_numpy(np.array(pyboy.screen_image()))
        pix = pix.permute(2,0,1)#.unsqueeze(0)
        pix = pix.float()

        pix = torch.nn.functional.avg_pool2d(pix, 2)
        pix = torch.nn.functional.avg_pool2d(pix, 2)
        pix = torch.nn.functional.avg_pool2d(pix, 2)
        pix = pix.view(-1).unsqueeze(0)
        print(pix.size())
        feats = pix.detach().numpy()

        '''
        pix = preprocess(pix).unsqueeze(0)
        #print('dims: ' + str(pix.size()))
        #print('type: ' + str(pix.type()))
        output = res34(pix)#[0]
        probs = torch.nn.functional.softmax(output[0], dim=0)
        probs = probs.unsqueeze(0)
        feats = probs.detach().numpy()
        '''

        if (len(all_stored_frame_vecs) == 0):
            # Element insertion (can be called several times):
            p.add_items(feats, np.array([len(all_stored_frame_vecs)]))
            all_stored_frame_vecs.append(feats)

        # Query dataset, k - number of closest elements
        labels, distances = p.knn_query(feats[0], k = 1)
        print(str(len(all_stored_frame_vecs)) + ' total frames indexed, current closest is: ' + str(distances[0]))
        if (distances[0] > 1500000.0): #0.0001):
            p.add_items(feats, np.array([len(all_stored_frame_vecs)]))
            all_stored_frame_vecs.append(feats)

        #print(feats)
    #    break

    frame += 1

#tile_map = pyboy.get_window_tile_map() # Get the TileView object for the window.

#index_map = tile_map.get_tile_matrix()

# For unattended use, the screen buffer can be displayed using the following:
#pyboy.get_screen_image().show()
