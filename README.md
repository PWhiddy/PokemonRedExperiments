

<img src="/assets/poke_map.gif?raw=true">

# Pokemon Red RL

Experiments training reinforcement learning agents to play Pokemon Red.   
Watch the [Video on Youtube!](https://youtube.com/the-video)

### To run the pretrained model locally:

1. Copy your Pokemon Red ROM into the base directory. It should be named `PokemonRed.gb` and will be 1MB. You can find this using google.
2. Move into the `baselines/` directory
3. Install dependencies:  
```pip install -r requirements.txt```  
It may be necessary to install the extra libraries [extra things]
4. Run:  
```python run_pretrained_interactive.py```

Interact with the emulator using the arrow keys and the `a` and `s` keys (A and B buttons).  
You can pause the AI's input during the game by editing `agent_enabled.txt`

### To train the model (requires a lot of cpu cores and memory):

1. Previous steps 1-3
2. Run  
```python run_baseline_parallel.py```

### Extra
Map visualization code can be found in `visualization/` directory.

