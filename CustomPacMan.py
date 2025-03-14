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
        self.pills = [] # Pill positions
        self.food = [] # food positions
        self.pacman_pos = () #pacman position
        self.pill_start_duration = 100
        self.pill_duration = 0 #amount of moves before pill goes unactive
        self.pill_active = False
        self.algo = "QL"
        self.lives = 1
        self.score = 0
        self.food_reward = 100
        self.eat_ghost_reward = 200
        self.pickup_pill_reward = 50
        self.lose_live_reward = -1000

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

        #Q-Learning Parameters
        self.alpha = 0.1 # Learning rate
        self.gamma = 0.9 # Discount factor
        self.epsilon = 0.1 # Exploration rate
        self.q_table = {}  # Dictionary to store Q-values



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

        #Place PacMan
        if(empty_cells):
                x, y = random.choice(empty_cells)
                self.grid[x, y] = 5 #Add PacMan (5) to grid
                empty_cells.remove((x, y)) #Remove empty cell from list
                self.pacman_pos = (x,y) #Add pacman position to array

        #Place Ghosts
        for i in range(self.ghost_count):
            if(empty_cells):
                x, y = random.choice(empty_cells)
                #self.grid[x, y] = 3 #Add ghost (3) to grid
                self.ghosts.append((x, y)) #Add ghost position to array

        #Place food in left empty spaces
        for x, y in empty_cells:
            self.grid[x, y] = 4 #Add Food (4) to grid
            self.food.append((x,y))

    def movePacMan(self, current_PacMan_pos, new_pos, old_ghost_positions):
        reward = 0
        if(self.grid[new_pos] == 4):#moved to food
            self.food_count -= 1
            self.food.remove((new_pos))
            self.score += self.food_reward
            reward = self.food_reward
        elif(self.grid[new_pos] == 2):#moved to pill
            self.pill_active = True
            self.pill_duration = self.pill_start_duration
            self.pill_count -= 1
            self.pills.remove(new_pos)
            self.score += self.pickup_pill_reward
            reward = self.pickup_pill_reward

        #check ghost interactions
        for i in range(self.ghost_count):
            if ((self.ghosts[i] == new_pos) or (new_pos == old_ghost_positions[i] and current_PacMan_pos == self.ghosts[i])):
                if(self.pill_active == True):
                    self.respawnGhost(i)
                    self.score += self.eat_ghost_reward
                    reward += self.eat_ghost_reward
                else:
                    self.lives -= 1
                    reward += self.lose_live_reward


        self.grid[self.pacman_pos] = 0 #Clear field
        self.pacman_pos = new_pos
        self.grid[new_pos] = 5 # set to PacMan

        return reward

    def chooseActionPacMan(self, use_greedy_strategy):
        if(self.algo == "random"):
            x, y = self.pacman_pos
            valid_moves = []
            for direction_x, direction_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  #up, down, left, right
                next_x, next_y = x + direction_x, y + direction_y
                if self.grid[next_x, next_y] != 1: #not wall
                    valid_moves.append((next_x, next_y))

            return random.choice(valid_moves), (direction_x, direction_y)

        elif(self.algo == "QL"):
            new_pos, action = self.QLearningAction(use_greedy_strategy)
            return new_pos, action
        elif(self.algo == "QRM"):
            return

    def QLearningAction(self, use_greedy_strategy):
        x, y = self.pacman_pos
        state = self.get_state()

        if not use_greedy_strategy:
            #explore
            if random.random() < self.epsilon:
                valid_moves = []
                for action, (direction_x, direction_y) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    next_x, next_y = x + direction_x, y + direction_y
                    if self.grid[next_x, next_y] != 1:  # Not a wall
                        valid_moves.append((action, (next_x, next_y)))

                chosen_action, next_pos = random.choice(valid_moves)
                return next_pos, self.action_to_direction(chosen_action)
            
        #exploit    
        valid_moves = []
        for action, (direction_x, direction_y) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            next_x, next_y = x + direction_x, y + direction_y
            if self.grid[next_x, next_y] != 1:  # Not a wall
                valid_moves.append((action, (next_x, next_y)))

        best_action, next_pos = max(valid_moves, key=lambda move: self.q_table.get((state, move[0]), 0))#check Q-value for every move in valid_moves. get max q-value (state, move[0]) where move[0] is the action and 0 is default value if not in dictionary yet.
        return next_pos, self.action_to_direction(best_action)

    def action_to_direction(self, action):
        if action == 0:  # UP
            return (-1, 0)
        elif action == 1:  # DOWN
            return (1, 0)
        elif action == 2:  # LEFT
            return (0, -1)
        elif action == 3:  # RIGHT
            return (0, 1)
        
    def updateQTable(self, state, action, reward, next_state):
        q_current = self.q_table.get((state, action), 0) # Get current Q-value. 0 as default value if not in dictionary yet
        q_next_max = max(self.q_table.get((next_state, a), 0) for a in range(4)) # Max Q-value of next state
        q_update = reward + self.gamma * q_next_max
        self.q_table[(state, action)] = q_current + self.alpha * (q_update - q_current) # Update q-table

    def get_state(self):
        x, y = self.pacman_pos
        ghost_positions = tuple(ghostPos for ghostPos in self.ghosts) # Store all ghost positions
        food_positions = tuple(foodPos for foodPos in self.food)
        pill_positions = tuple(pillPos for pillPos in self.pills)
        pill_status = int(self.pill_active) # 1 if active, 0 otherwise
        pill_timer = self.pill_duration if self.pill_active else 0

        return (x, y) + ghost_positions + pill_positions + food_positions + (pill_status, pill_timer)

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


    def step(self, use_greedy_strategy):
        
        #store current PacMan Position
        current_PacMan_pos = self.pacman_pos

        #pacman action
        new_pos_pacMan, action = self.chooseActionPacMan(use_greedy_strategy)#choose action

        #old ghosts positions
        old_ghost_positions = []
        for i in range(self.ghost_count):
            ghost_x, ghost_y = self.ghosts[i]
            old_ghost_positions.append((ghost_x, ghost_y))

        #ghosts actions
        for i in range(self.ghost_count):
            ghost_x, ghost_y = self.ghosts[i]
            new_pos = self.ghostSemiRandomMove((ghost_x, ghost_y), 0.7)
            self.ghosts[i] = (new_pos[0], new_pos[1])

        #move PacMan & store reward
        reward = self.movePacMan(current_PacMan_pos, new_pos_pacMan, old_ghost_positions)

        self.updateQTable(state=current_PacMan_pos, action=action, reward=reward, next_state=self.pacman_pos)

        if(self.pill_active):
            #store current PacMan Position
            current_PacMan_pos = self.pacman_pos

            new_pos_pacMan, action = self.chooseActionPacMan(use_greedy_strategy)#choose extra action

            #move PacMan & store reward
            reward = self.movePacMan(current_PacMan_pos, new_pos_pacMan, old_ghost_positions)#do extra action

            self.updateQTable(state=current_PacMan_pos, action=action, reward=reward, next_state=self.pacman_pos)

            self.pill_duration -= 1
            if(self.pill_duration <= 0):
                self.pill_active = False

        if(self.lives <= 0):
            self.play = False
            print("Out of lives")
            self.reset()
        elif(self.food_count <= 0):
            self.play = False
            print("Eaten all food")
            self.reset()

        
    def reset(self):
        self.grid = self.loadMaze()  # Reload maze
        self.lives = 1  # Reset lives
        self.score = 0  # Reset score
        self.play = True
        self.pill_count = 4 #amount of pills in the maze
        self.ghost_count = 4 #amount of ghosts in the maze
        self.ghosts = [] #ghost positions
        self.pills = [] #pill positions
        self.food = [] # food positions
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
                elif self.grid[x, y] == 2: #Pill
                    pygame.draw.ellipse(screen, (0, 255, 0), (y * cell_size + cell_size/4, x * cell_size +cell_size/4, cell_size/2, cell_size/2))
                elif self.grid[x, y] == 4: #food
                    pygame.draw.ellipse(screen, (0, 255, 255), (y * cell_size + cell_size/4, x * cell_size +cell_size/4, cell_size/2, cell_size/2))
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
