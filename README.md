# TFG - Pokémon Red RL Agent with Segmentation

Final Degree Project for the Computer Engineering program at the University of the Balearic Islands.  
This project is based on and adapted from the [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) repository by Peter Whidden.  
It introduces a segmentation model that simplifies the game environment, helping the reinforcement learning agent better interpret and navigate the world.

---

## 🛠️ Installation & Usage

> **Recommended Python version:** `Python 3.10+`  
> It's highly suggested to use [Anaconda](https://anaconda.org/anaconda/python) for easier environment management.

### Setup Instructions

1. Place your own copy of **Pokémon Red** in the folder:  
   `Game&Results/Pokemon_Red_Env`

2. Install required dependencies:  
   ```bash
   pip install -r requirements.txt
   ````

3. You're all set!
To start training, run one of the following commands:
- **Standard training**:
     ```bash
       python train.py
     ````
- **Endless training loop** (until manually stopped):
     ```bash
       non_stop_train.bat
     ````
  ✨ **Tip**: Before training, check and adjust the settings in *config.yml* to suit your preferences.

## 📊 Visualization
You can follow the live training progress through the Pokémon Red map visualizer by Peter Whidden: [Pokemon Red Visualizer](https://pwhiddy.github.io/pokerl-map-viz/)

## 📚 References
his project is an adaptation of Peter Whidden's work.
You can find the original repository and detailed documentation here:
👉 [PokemonRedExperiments on GitHub](https://github.com/PWhiddy/PokemonRedExperiments)
