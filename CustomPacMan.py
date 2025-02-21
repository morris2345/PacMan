import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class CustomPacManEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, maze_size="normal"):
        super().__init__()

        self.pill_amount = 4 #amount of pills in the maze
        self.ghost_amount = 4 #amount of ghosts in the maze
        self.ghosts = [] #ghost positions
        self.pills = [] #pill positions
        self.pacman #pacman position

        # Define available maze files
        self.maze_files = {
            "small": "mazes/small.txt",
            "normal": "mazes/normal.txt",#normal size = 28x31
            "large": "mazes/large.txt",
        }

        self.maze_size = maze_size
        self.grid = self.loadMaze()  # Load the maze from file
        self.fillMaze()#Fill maze with Pills, Ghosts & Food

        # Define action space (UP, DOWN, LEFT, RIGHT)
        self.action_space = spaces.Discrete(4)


    #Load the maze file containing walls & empty spaces
    def loadMaze(self):
        fileName = self.maze_files.get(self.maze_size, "mazes/normal.txt") #load maze file, default normal

        with open(fileName, "r") as file:
            lines = file.readlines()

        maze = np.array([[int(char) for char in line.strip()] for line in lines], dtype=np.int8)
        return maze
    
    #Fill the maze with Pills, Ghosts & Food
    def fillMaze(self):
        empty_cells = list(zip(*np.where(self.grid == 0)))

        #Place Pills
        for i in range(self.pill_amount):
            if(empty_cells):
                x, y = random.choice(empty_cells)
                self.grid[x, y] = 2 #Add Pill (2) to grid
                empty_cells.remove((x, y)) #Remove empty cell from list
                self.pills.append((x,y)) #Add pill position to array

        #Place Ghosts
        for i in range(self.ghost_amount):
            if(empty_cells):
                x, y = random.choice(empty_cells)
                self.grid[x, y] = 3 #Add ghost (3) to grid
                self.ghosts.append((x, y)) #Add ghost position to array
                empty_cells.remove((x, y)) #Remove empty cell from list

        #Place PacMan
        if(empty_cells):
                x, y = random.choice(empty_cells)
                self.grid[x, y] = 5 #Add PacMan (5) to grid
                empty_cells.remove((x, y)) #Remove empty cell from list
                self.pacman = (x,y) #Add pacman position to array

        #Place food in left empty spaces
        for x, y in empty_cells:
            self.grid[x, y] = 4 #Add Food (4) to grid

        #print(self.pills)
        #print(self.ghosts)
