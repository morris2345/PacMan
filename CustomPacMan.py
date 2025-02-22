import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque

class CustomPacManEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, maze_size="normal"):
        super().__init__()

        self.pill_amount = 4 #amount of pills in the maze
        self.ghost_amount = 4 #amount of ghosts in the maze
        self.ghosts = [] #ghost positions
        self.pills = [] #pill positions
        self.pacman = () #pacman position
        self.pill_duration = 30
        self.pill_active = True

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


    def bfsPathFinding(self, ghost_pos):
        queue = deque([(ghost_pos, [])]) #create dubble ended queue with current pos and path
        visited = set()

        while queue:
            (x, y), path = queue.popleft()

            #check if on pacman, return first step of path
            if(x, y) == self.pacman:
                return path[0]
            
            for direction_x, direction_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  #up, down, left, right
                next_x, next_y = x + direction_x, y + direction_y

                if(next_x, next_y) not in visited and self.grid[next_x,next_y] != 1: #not visited & not wall
                    queue.append(((next_x, next_y), path + [(next_x, next_y)]))
                    visited.add((next_x,next_y))


    def ghostSemiRandomMove(self, ghost_pos, chase_prob):
        x, y = ghost_pos

        if self.pill_active == True: #If PacMan can eat ghosts
            valid_moves = []
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            bfs_location = self.bfsPathFinding(ghost_pos) #Returns the best location to move to when going towards PacMan
            bfs_move = (bfs_location[0] - x, bfs_location[1] - y) #Returns the best move (up, down, left, right) when going towards PacMan
            moves.remove(bfs_move) #Remove best move for going towards PacMan, because ghost is running away

            for direction_x, direction_y in moves:
                next_x, next_y = x + direction_x, y + direction_y
                if self.grid[next_x, next_y] != 1: #not wall
                    valid_moves.append((next_x, next_y))

            return random.choice(valid_moves)

        else: #If PacMan can get eaten
            if random.uniform(0,1) <= chase_prob:
                print('test')
                return self.bfsPathFinding(ghost_pos) #Move with shortest path towards PacMan
            else:
                valid_moves = []

                for direction_x, direction_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  #up, down, left, right
                    next_x, next_y = x + direction_x, y + direction_y
                    if self.grid[next_x, next_y] != 1: #not wall
                        valid_moves.append((next_x, next_y))

                return random.choice(valid_moves)


    def step(self):
        
        #pacman action

        for i in range(self.ghost_amount):
            ghost_pos = self.ghosts[i]
            new_pos = self.ghostSemiRandomMove(ghost_pos, 0.7)
            self.grid[ghost_pos] = 0 # clear ghost                           !!! still need to change back to food/pill if it was there !!!
            self.ghosts[i] = new_pos
            self.grid[new_pos] = 3 #set to ghost
