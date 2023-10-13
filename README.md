# Pokemon Red RL

<a href="https://youtu.be/DcYLT37ImBY">
  <img src="/assets/poke_map.gif?raw=true">
</a>

Experiments training reinforcement learning agents to play Pokemon Red.
Watch the [Video on Youtube!](https://youtu.be/DcYLT37ImBY)  

<a href="https://youtu.be/DcYLT37ImBY">
  <img src="/assets/Pokemon YT5 FFFFinal.jpg?raw=true" width="512">
</a>
  
### To run the pretrained model locally

1. Copy your legally obtained Pokemon Red ROM into the base directory. You can find this using google, it should be 1MB. Rename it to `PokemonRed.gb` if it is not already. 
2. Move into the `baselines/` directory:  
 ```cd baselines```
3. Install dependencies:  
```pip install -r requirements.txt```  
It may be necessary in some cases to separately install the SDL libraries.
4. Run:  
```python run_pretrained_interactive.py```

Interact with the emulator using the arrow keys and the `a` and `s` keys (A and B buttons).  
You can pause the AI's input during the game by editing `agent_enabled.txt`

Note that the Pokemon.gb file MUST be in the main directory and your current directory MUST be the `baselines/` directory in order for this to work.

### To train the model (requires a lot of cpu cores and memory):

1. Previous steps 1-3
2. Run:  
```python run_baseline_parallel.py```

### Tracking training progress

You can view the current state of each emulator, plot basic stats, and compare to previous runs using the `VisualizeProgress.ipynb` notebook.

### Extra

Map visualization code can be found in `visualization/` directory.
