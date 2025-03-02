import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque

class CustomPacManEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, maze_size="normal"):
        super().__init__()

        self.pill_count = 4 #amount of pills in the maze
        self.ghost_count = 4 #amount of ghosts in the maze
        self.ghosts = [] #ghost positions
        self.pills = [] #pill positions
        self.pacman_pos = () #pacman position
        self.pill_start_duration = 10
        self.pill_duration = 10#amount of moves before pill goes unactive
        self.pill_active = False
        self.algo = "random"
        self.lives = 1
        self.score = 0
        self.food_reward = 100
        self.eat_ghost_reward = 200

        # Define available maze files
        self.maze_files = {
            "small": "mazes/small.txt",
            "normal": "mazes/normal.txt",#normal size = 28x31
            "large": "mazes/large.txt",
        }

        self.maze_size = maze_size
        self.grid = self.loadMaze()  # Load the maze from file
        self.fillMaze()#Fill maze with Pills, Ghosts & Food

        self.food_count = np.count_nonzero(self.grid == 4) #amount of food in the maze

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
        for i in range(self.pill_count):
            if(empty_cells):
                x, y = random.choice(empty_cells)
                self.grid[x, y] = 2 #Add Pill (2) to grid
                empty_cells.remove((x, y)) #Remove empty cell from list
                self.pills.append((x,y)) #Add pill position to array

        #Place Ghosts
        for i in range(self.ghost_count):
            if(empty_cells):
                x, y = random.choice(empty_cells)
                self.grid[x, y] = 3 #Add ghost (3) to grid
                self.ghosts.append((x, y, 4)) #Add ghost position to array
                empty_cells.remove((x, y)) #Remove empty cell from list

        #Place PacMan
        if(empty_cells):
                x, y = random.choice(empty_cells)
                self.grid[x, y] = 5 #Add PacMan (5) to grid
                empty_cells.remove((x, y)) #Remove empty cell from list
                self.pacman_pos = (x,y) #Add pacman position to array

        #Place food in left empty spaces
        for x, y in empty_cells:
            self.grid[x, y] = 4 #Add Food (4) to grid

        #print(self.pills)
        #print(self.ghosts)

    def movePacMan(self, new_pos):
        if(self.grid[new_pos] == 4):#moved to food
            self.food_count -= 1
            self.score += self.food_reward
        elif(self.grid[new_pos == 2]):#moved to pill
            self.pill_active = True
            self.pill_duration = self.pill_start_duration
            self.pill_count -= 1
        elif(self.grid[new_pos == 3]):#move to ghost
            if(self.pill_active):
                #eat ghost/respawn ghost
                self.score += self.eat_ghost_reward
            else:
                self.lives -= 1


        self.grid[self.pacman_pos] = 0 #Clear field
        self.pacman_pos = new_pos
        self.grid[new_pos] = 5 # set to PacMan

    def chooseActionPacMan(self):
        if(self.algo == "random"):
            x, y = self.pacman_pos
            valid_moves = []
            for direction_x, direction_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  #up, down, left, right
                next_x, next_y = x + direction_x, y + direction_y
                if self.grid[next_x, next_y] != 1: #not wall
                    valid_moves.append((next_x, next_y))

            return random.choice(valid_moves)

        elif(self.algo == "QL"):
            return
        elif(self.algo == "QRM"):
            return


    def bfsPathFinding(self, ghost_pos):
        queue = deque([(ghost_pos, [])]) #create dubble ended queue with current pos and path
        visited = set()

        while queue:
            (x, y), path = queue.popleft()

            #check if on pacman, return first step of path
            if(x, y) == self.pacman_pos:
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
        new_pos = self.chooseActionPacMan()#choose action

        #ghosts actions
        for i in range(self.ghost_count):
            ghost_x, ghost_y, previous_value = self.ghosts[i]
            new_pos = self.ghostSemiRandomMove((ghost_x, ghost_y), 0.7)
            self.grid[ghost_x, ghost_y] = previous_value # clear ghost
            self.ghosts[i] = (new_pos[0], new_pos[1], self.grid[new_pos])
            self.grid[new_pos] = 3 #set to ghost

        #move PacMan
        self.movePacMan(self, new_pos)
        if(self.pill_active):
            self.pill_duration -= 1
            new_pos = self.chooseActionPacMan()#choose extra action
            self.movePacMan(self, new_pos)#do extra action
            if(self.pill_duration <= 0):
                self.pill_active = False

        if(self.lives <= 0):
            #lose
            return
        elif(self.food_count <= 0):
            #win
            return
