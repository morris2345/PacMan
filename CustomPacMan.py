import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomPacManEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, maze_size="normal"):
        super().__init__()

        # Define available maze files
        self.maze_files = {
            "small": "mazes/small.txt",
            "normal": "mazes/normal.txt",#normal size = 28x31
            "large": "mazes/large.txt",
        }

        self.maze_size = maze_size
        self.grid = self.loadMaze()  # Load the maze from file

        # Define action space (UP, DOWN, LEFT, RIGHT)
        self.action_space = spaces.Discrete(4)

    def loadMaze(self):
        fileName = self.maze_files.get(self.maze_size, "mazes/normal.txt") #load maze file, default normal

        with open(fileName, "r") as file:
            lines = file.readlines()

        maze = np.array([[int(char) for char in line.strip()] for line in lines], dtype=np.int8)
        return maze