import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
import math
import pygame

class CustomPacManEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, maze_size="normal"):
        super().__init__()

        self.play = True
        self.pill_count = 4 #amount of pills in the maze
        self.ghost_count = 4 #amount of ghosts in the maze
        self.ghosts = [] #ghost positions
        self.pills = [] #pill positions
        self.pacman_pos = () #pacman position
        self.pill_start_duration = 100
        self.pill_duration = 100#amount of moves before pill goes unactive
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
                #self.grid[x, y] = 3 #Add ghost (3) to grid
                self.ghosts.append((x, y)) #Add ghost position to array
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

    def movePacMan(self, new_pos):
        if(self.grid[new_pos] == 4):#moved to food
            self.food_count -= 1
            self.score += self.food_reward
        elif(self.grid[new_pos] == 2):#moved to pill
            self.pill_active = True
            self.pill_duration = self.pill_start_duration
            self.pill_count -= 1

        #check ghost interactions
        for i in range(self.ghost_count):
            ghost_x, ghost_y = self.ghosts[i]
            pac_x, pac_y = new_pos
            if ((int(ghost_x), int(ghost_y)) == (int(pac_x), int(pac_y))):
                if(self.pill_active == True):
                    self.respawnGhost(i)
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
                return self.bfsPathFinding(ghost_pos) #Move with shortest path towards PacMan
            else:
                valid_moves = []

                for direction_x, direction_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  #up, down, left, right
                    next_x, next_y = x + direction_x, y + direction_y
                    if self.grid[next_x, next_y] != 1: #not wall
                        valid_moves.append((next_x, next_y))

                return random.choice(valid_moves)

    def respawnGhost(self, ghost_index):
        distance = 10

        empty_cells = list(zip(*np.where((self.grid == 0) | (self.grid == 4) | (self.grid == 2))))
        
        valid_cells = [cell for cell in empty_cells if (math.dist(self.pacman_pos, self.ghosts[ghost_index]) <= distance)]

        if(valid_cells == None):
            print("No cell found to respawn ghost")
        else:
            new_ghost_x, new_ghost_y = random.choice(valid_cells)
            self.ghosts[ghost_index] = (new_ghost_x, new_ghost_y)


    def step(self):
        
        #pacman action
        new_pos_pacMan = self.chooseActionPacMan()#choose action

        #ghosts actions
        for i in range(self.ghost_count):
            ghost_x, ghost_y = self.ghosts[i]
            new_pos = self.ghostSemiRandomMove((ghost_x, ghost_y), 0.7)
            self.ghosts[i] = (new_pos[0], new_pos[1])

        #move PacMan
        self.movePacMan(new_pos_pacMan)
        if(self.pill_active):
            self.pill_duration -= 1
            new_pos_pacMan = self.chooseActionPacMan()#choose extra action
            self.movePacMan(new_pos_pacMan)#do extra action
            if(self.pill_duration <= 0):
                self.pill_active = False

        if(self.lives <= 0):
            self.play = False
            print("Out of lives")
            return
        elif(self.food_count <= 0):
            self.play = False
            print("Eaten all food")
            return
        
    def reset(self):
        self.grid = self.loadMaze()  # Reload maze
        self.lives = 1  # Reset lives
        self.score = 0  # Reset score
        self.play = True
        self.pill_count = 4 #amount of pills in the maze
        self.ghost_count = 4 #amount of ghosts in the maze
        self.ghosts = [] #ghost positions
        self.pills = [] #pill positions
        self.pacman_pos = () #pacman position
        self.pill_duration = 0 #amount of moves before pill goes unactive
        self.pill_active = False
        self.fillMaze()  # Fill maze again with Pills, Ghosts, Food & PacMan
        self.food_count = np.count_nonzero(self.grid == 4)  # Amount of food
        #return self.pacman_pos
        
    def render(self):
        cell_size = 40
        screen = pygame.display.set_mode((self.grid.shape[1] * cell_size, self.grid.shape[0] * cell_size))
        screen.fill((0, 0, 0))

        # Draw the Maze
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                if self.grid[x, y] == 1:  # Wall
                    pygame.draw.rect(screen, (0, 0, 255), (y * cell_size, x * cell_size, cell_size, cell_size))
                else:  # Empty space (Paths)
                    pygame.draw.rect(screen, (0, 0, 0), (y * cell_size, x * cell_size, cell_size, cell_size))

        # Pac-Man
        pac_x, pac_y = self.pacman_pos
        pygame.draw.circle(screen, (255, 255, 0), (pac_y * cell_size + cell_size / 2, pac_x * cell_size + cell_size / 2), cell_size / 2)

        # Ghosts
        for ghost_x, ghost_y in self.ghosts:
            pygame.draw.polygon(screen, (255, 0, 0), [
                (ghost_y * cell_size, ghost_x * cell_size + cell_size),  # Bottom-left
                (ghost_y * cell_size + cell_size, ghost_x * cell_size + cell_size),  # Bottom-right
                (ghost_y * cell_size + cell_size / 2, ghost_x * cell_size)  # Top-center
            ])

        pygame.display.flip()
