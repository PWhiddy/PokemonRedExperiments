import sys
from pyboy import PyBoy, windowevent
import torch
import torchvision

pyboy = PyBoy(
        "./PokemonRed.gb",
        debugging=False,
        disable_input=True,
        # window_type="headless", # For unattended use, for example machine learning
        window_type='SDL2',
        hide_window="--quiet" in sys.argv,
    )
pyboy.set_emulation_speed(0)

preprocess = torchvision.transforms.Compose([
#    torchvision.transforms.Resize(256),
#    torchvision.transforms.CenterCrop(224),
#    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

frame = 0
while not pyboy.tick():

    if frame%200 == 0:
        print('pressing start...')
        pyboy.send_input(windowevent.PRESS_BUTTON_START)
    else:
        pyboy.send_input(windowevent.RELEASE_BUTTON_START)

    if frame%5 == 0:
        print('pressing A...')
        pyboy.send_input(windowevent.PRESS_BUTTON_A)
    else:
        pyboy.send_input(windowevent.RELEASE_BUTTON_A)

    if frame % 10 == 0:
        #print(pyboy.get_raw_screen_buffer())
        pix = torch.from_numpy(pyboy.get_screen_ndarray())
        pix = pix.permute(2,0,1)#.unsqueeze(0)
        pix = pix.float()
        pix = preprocess(pix).unsqueeze(0)
        print('dims: ' + str(pix.size()))
        print('type: ' + str(pix.type()))
    #    break

    frame += 1

#tile_map = pyboy.get_window_tile_map() # Get the TileView object for the window.

#index_map = tile_map.get_tile_matrix()

# For unattended use, the screen buffer can be displayed using the following:
#pyboy.get_screen_image().show()
