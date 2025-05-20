# Train RL agents to play Pokemon Red

### New 10-19-24! Updated & Simplified V2 Training Script - See V2 below
### New 1-29-24! - [Multiplayer Live Training Broadcast](https://github.com/pwhiddy/pokerl-map-viz/)  🎦 🔴 [View Here](https://pwhiddy.github.io/pokerl-map-viz/)
Stream your training session to a shared global game map using the [Broadcast Wrapper](/baselines/stream_agent_wrapper.py)  

See how in [Training Broadcast](#training-broadcast) section
  
## Watch the Video on Youtube! 

<p float="left">
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/youtube.jpg?raw=true" height="192">
  </a>
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/poke_map.gif?raw=true" height="192">
  </a>
</p>

## Join the discord server
[![Join the Discord server!](https://invidget.switchblade.xyz/RvadteZk4G)](http://discord.gg/RvadteZk4G)
  
## Running the Pretrained Model Interactively 🎮  
🐍 Python 3.10+ is recommended. Other versions may work but have not been tested.   
You also need to install ffmpeg and have it available in the command line.

### Windows Setup
Refer to this [Windows Setup Guide](windows-setup-guide.md)

### For AMD GPUs
Follow this [guide to install pytorch with ROCm support](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/howto_wsl.html)

### Linux / MacOS

V2 is now recommended over the original version. You may follow all steps below but replace `baselines` with `v2`.

1. Copy your legally obtained Pokemon Red ROM into the base directory. You can find this using google, it should be 1MB. Rename it to `PokemonRed.gb` if it is not already. The sha1 sum should be `ea9bcae617fdf159b045185467ae58b2e4a48b9a`, which you can verify by running `shasum PokemonRed.gb`. 
2. Move into the `baselines/` directory:  
 ```cd baselines```  
3. Install dependencies:  
```pip install -r requirements.txt```  
It may be necessary in some cases to separately install the SDL libraries.
MacOS users will need to use ```macos_requirements.txt``` instead of ```requirements.txt```
4. Run:  
```python run_pretrained_interactive.py```
  
Interact with the emulator using the arrow keys and the `a` and `s` keys (A and B buttons).  
You can pause the AI's input during the game by editing `agent_enabled.txt`

Note: the Pokemon.gb file MUST be in the main directory and your current directory MUST be the `baselines/` directory in order for this to work.

## Training the Model 🏋️ 

<img src="/assets/grid.png?raw=true" height="156">


### V2

- Trains faster and with less memory
- Reaches Cerulean
- Streams to map by default
- Other improvements

Replaces the frame KNN with a coordinate based exploration reward, as well as some other tweaks.
1. Previous steps but in the `v2` directory instead of `baselines`
2. Run:
```python baseline_fast_v2.py```

## Tracking Training Progress 📈

### Training Broadcast
Stream your training session to a shared global game map using the [Broadcast Wrapper](/baselines/stream_agent_wrapper.py) on your environment like this:
```python
env = StreamWrapper(
            env, 
            stream_metadata = { # All of this is part is optional
                "user": "super-cool-user", # choose your own username
                "env_id": id, # environment identifier
                "color": "#0033ff", # choose your color :)
                "extra": "", # any extra text you put here will be displayed
            }
        )
```

Hack on the broadcast viewing client or set up your own local stream with this repo:  
  
https://github.com/pwhiddy/pokerl-map-viz/

### Local Metrics
The current state of each game is rendered to images in the session directory.   
You can track the progress in tensorboard by moving into the session directory and running:  
```tensorboard --logdir .```  
You can then navigate to `localhost:6006` in your browser to view metrics.  
To enable wandb integration, change `use_wandb_logging` in the training script to `True`.

## Static Visualization 🐜
Map visualization code can be found in `visualization/` directory.

## Supporting Libraries
Check out these awesome projects!
### [PyBoy](https://github.com/Baekalfen/PyBoy)
<a href="https://github.com/Baekalfen/PyBoy">
  <img src="/assets/pyboy.svg" height="64">
</a>

### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
<a href="https://github.com/DLR-RM/stable-baselines3">
  <img src="/assets/sblogo.png" height="64">
</a>
