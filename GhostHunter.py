import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
import math
import pygame

class GhostHunterEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, maze_size="small", algo="QL"):
        super().__init__()

        self.play = True #game running or not
        self.won = 0 #game won or not (1=won, 0=lost)
        self.start_pill_count = 2 #amount of pills in the maze at the start of the game
        self.pill_count = 2 #amount of pills in the maze
        self.ghost_count = 1 #amount of ghosts in the maze
        self.ghosts = [] #ghost positions
        self.pills = [] # Pill positions
        self.food = [] # food positions
        self.pacman_pos = () #pacman position
        self.pill_start_duration = 50 #starting duration of pills
        self.pill_duration = 0 #amount of moves left before pill goes unactive
        self.pill_active = False #if pill active or not
        self.algo = algo #what type of agent (QL or QL with RM)
        self.version = "Ghost_Hunter" #Normal/Vegan/Ghost Hunter/Speedrun
        self.lives = 1
        self.score = 0  #achieved game score
        self.max_food = 4 #max food in the maze
        self.ghosts_eaten = 0 #ghosts eaten this game

        #rewards given to agent
        self.food_reward = 100 #pick up food
        self.eat_ghost_reward = 500 #eat ghost
        self.pickup_pill_reward = 200 #Pick up Pill
        self.lose_live_reward = -1000 #Eaten by ghost
        self.step_reward = -5 #every step
        self.win_reward = 1000 #game won (all food collected)

        self.rm_state = 0 #reward machine state (doesnt get updated if agent uses QL without RM)

        # Define available maze files
        self.maze_files = {
            "small": "mazes/smallGhostHunterMaze.txt", #small size = 15x15
            "normal": "mazes/normal.txt",#normal size = 28x31
            "large": "mazes/large.txt",
        }

        self.maze_size = maze_size
        self.grid = self.loadMaze()  # Load the maze from file
        self.fillMaze()#Fill maze with Pills, Ghosts & Food

        self.food_count = np.count_nonzero(self.grid == 4) #amount of food in the maze
        # Define action space (UP, DOWN, LEFT, RIGHT)
        #self.action_space = spaces.Discrete(4)

        #Q-Learning Parameters
        self.alpha = 0.05 # Learning rate
        self.gamma = 0.99 # Discount factor
        self.epsilon = 1.0 # Exploration rate
        self.q_table = {}  # Dictionary to store Q-values



    #Load the maze file containing walls & empty spaces
    def loadMaze(self):
        fileName = self.maze_files.get(self.maze_size, "mazes/normal.txt") #load maze file, default normal

        with open(fileName, "r") as file:
            lines = file.readlines()

        maze = np.array([[int(char) for char in line.strip()] for line in lines], dtype=np.int8)#create a numpy 2D array of the grid
        return maze
    
    #Fill the maze with Pills, Ghosts & Food
    def fillMaze(self):
        empty_cells = list(zip(*np.where(self.grid == 0)))#squares where grid is empty squares (zeros)
        self.pills = list(zip(*np.where(self.grid == 2)))
        self.food = list(zip(*np.where(self.grid == 4)))

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
                self.ghosts.append((x, y)) #Add ghost position to list

    #move PacMan to new location, returns reward for agent
    def movePacMan(self, current_PacMan_pos, new_pos, old_ghost_positions):
        reward = 0
        reward += self.step_reward
        if(self.grid[new_pos] == 4):#moved to food
            self.food_count -= 1
            self.food.remove((new_pos))
            self.score += 100 #add to game score
            reward += self.food_reward # add reward for food pickup
        elif(self.grid[new_pos] == 2):#moved to pill
            self.pill_active = True # Activate Pill
            self.pill_duration = self.pill_start_duration #set pill duration
            self.pill_count -= 1
            self.pills.remove(new_pos) #remove pill from list of pill locations
            self.score += 200
            reward = self.pickup_pill_reward

        self.updateRMState() #update the reward machine state

        #check ghost interactions
        for i in range(self.ghost_count):
            if ((self.ghosts[i] == new_pos) or (new_pos == old_ghost_positions[i] and current_PacMan_pos == self.ghosts[i])): #if ghost on location of pacman or pacman and ghost moved through eachother
                if(self.pill_active == True):#if pill active eat ghost
                    self.respawnGhost(i) #respawn ghost 
                    self.score += 500
                    reward += self.eat_ghost_reward
                    self.ghosts_eaten += 1
                elif(self.pill_active == False): #if pill not active pacman gets eaten
                    self.lives -= 1
                    reward += self.lose_live_reward


        self.grid[self.pacman_pos] = 0 #Clear field
        self.pacman_pos = new_pos
        self.grid[new_pos] = 5 # set to PacMan

        return reward #return reward for agent

    #choose action based on greedy strategy and algorithm of agent
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
            new_pos, action = self.QRMAction(use_greedy_strategy)
            return new_pos, action

    #return position and action chosen by Q-Learning without Reward Machine
    def QLearningAction(self, use_greedy_strategy):
        x, y = self.pacman_pos #current position of pacman
        state = self.getState() #current state

        if not use_greedy_strategy:
            #explore
            if random.random() < self.epsilon:
                valid_moves = [] #moves that don't move pacman into a wall
                for action, (direction_x, direction_y) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    next_x, next_y = x + direction_x, y + direction_y #postion after action
                    if self.grid[next_x, next_y] != 1:  # Not a wall
                        valid_moves.append((action, (next_x, next_y))) #add to valid moves if pacman can move that way

                chosen_action, next_pos = random.choice(valid_moves) #choose random action that is possible
                return next_pos, self.actionToDirection(chosen_action)
            
        #exploit    
        valid_moves = []
        for action, (direction_x, direction_y) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            next_x, next_y = x + direction_x, y + direction_y
            if self.grid[next_x, next_y] != 1:  # Not a wall
                valid_moves.append((action, (next_x, next_y)))

        best_action, next_pos = max(valid_moves, key=lambda move: self.q_table.get((state, move[0]), 0))#check Q-value for every move in valid_moves. get max q-value (state, move[0]) where move[0] is the action and 0 is default value if not in dictionary yet.
        return next_pos, best_action

    #return direction tuple from a number
    def actionToDirection(self, action):
        if action == 0:  # UP
            return (-1, 0)
        elif action == 1:  # DOWN
            return (1, 0)
        elif action == 2:  # LEFT
            return (0, -1)
        elif action == 3:  # RIGHT
            return (0, 1)
        
    #update the q-table
    def updateQTable(self, state, action, reward, next_state):
        q_current = self.q_table.get((state, action), 0) # Get current Q-value. 0 as default value if not in dictionary yet
        q_next_max = max(self.q_table.get((next_state, a), 0) for a in range(4)) # Max Q-value of next state
        q_update = reward + self.gamma * q_next_max
        self.q_table[(state, action)] = q_current + self.alpha * (q_update - q_current) # Update q-table

    #return the state
    def getState(self):
        x, y = self.pacman_pos #current postition of PacMan
        
        #position of all ghosts
        ghost_positions = tuple(ghostPos for ghostPos in self.ghosts)

        #positions of all pills
        pill_positions = tuple(pillPos for pillPos in self.pills)

        #if pacman is empowered or not
        pill_status = int(self.pill_active) # 1 if active, 0 otherwise

        state = (x, y) + (ghost_positions , pill_positions , pill_status) # State of environment

        return (state, self.rm_state) # Return state + Reward Machine State. (RM State will always be 0 for normal QLearning)
    
    #updates Reward Machine state, which changes rewards
    def updateRMState(self):
        if(self.algo == "QRM"): #only update if using Q-Learning with Reward Machine
            if self.rm_state == 0 and self.pill_active: #If in non powered state and pill active
                self.rm_state = 1  #Transition to "Pill Active Mode"
                if(self.version == "Ghost_Hunter"):#change rewards
                    self.food_reward = 100
                    self.eat_ghost_reward = 500
                    self.pickup_pill_reward = -200 #picking up pill when already active is bad
                    self.lose_live_reward = -1000
                    self.step_reward = -5
                    self.win_reward = 1000

            elif self.rm_state == 1 and not self.pill_active: #if empowered but pill inactive
                self.rm_state = 0  #Transition back to normal
                if(self.version == "Ghost_Hunter"):#change rewards
                    self.food_reward = 100
                    self.eat_ghost_reward = 500
                    self.pickup_pill_reward = 200
                    self.lose_live_reward = -1000
                    self.step_reward = -5
                    self.win_reward = 1000
    
    #choose action for Q-Learning agent with Reward Machine
    def QRMAction(self, use_greedy_strategy):
        x, y = self.pacman_pos #current pacman position
        state = self.getState() #current state

        if not use_greedy_strategy:
            #explore
            if random.random() < self.epsilon:
                valid_moves = []
                for action, (direction_x, direction_y) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    next_x, next_y = x + direction_x, y + direction_y
                    if self.grid[next_x, next_y] != 1:  # Not a wall
                        valid_moves.append((action, (next_x, next_y)))

                chosen_action, next_pos = random.choice(valid_moves)
                return next_pos, self.actionToDirection(chosen_action)
            
        #exploit    
        valid_moves = []
        for action, (direction_x, direction_y) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            next_x, next_y = x + direction_x, y + direction_y
            if self.grid[next_x, next_y] != 1:  # Not a wall
                valid_moves.append((action, (next_x, next_y)))

        best_action, next_pos = max(valid_moves, key=lambda move: self.q_table.get((state, move[0]), 0))#check Q-value for every move in valid_moves. get max q-value (state, move[0]) where move[0] is the action and 0 is default value if not in dictionary yet.
        return next_pos, best_action

    #Breath first search pathfinding for Ghosts
    def bfsPathFinding(self, ghost_pos):
        queue = deque([(ghost_pos, [])]) #create dubble ended queue with current pos and path
        visited = set() #set of visited squares

        while queue:
            (x, y), path = queue.popleft()

            #check if on pacman, return first step of path
            if(x, y) == self.pacman_pos:
                return path[0] if path else ghost_pos #if path not empty return first move from path that lead to PacMan else return current position
            
            for direction_x, direction_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  #up, down, left, right
                next_x, next_y = x + direction_x, y + direction_y

                if(next_x, next_y) not in visited and self.grid[next_x,next_y] != 1: #not visited & not wall
                    queue.append(((next_x, next_y), path + [(next_x, next_y)]))
                    visited.add((next_x,next_y))

        return ghost_pos

    #Returns ghost move
    def ghostSemiRandomMove(self, ghost_pos, chase_prob):
        x, y = ghost_pos #current ghost position

        if self.pill_active == True: #If PacMan can eat ghosts
            valid_moves = []
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            bfs_location = self.bfsPathFinding(ghost_pos) #Returns the best location to move to when going towards PacMan
            bfs_move = (bfs_location[0] - x, bfs_location[1] - y) #Returns the best move (up, down, left, right) when going towards PacMan
            if bfs_move in moves:
                moves.remove(bfs_move) #Remove best move for going towards PacMan, because ghost is running away

            for direction_x, direction_y in moves: #check which moves are valid from left over directions
                next_x, next_y = x + direction_x, y + direction_y
                if self.grid[next_x, next_y] != 1: #not wall
                    valid_moves.append((next_x, next_y))

            return random.choice(valid_moves) #return one of the moves that isnt the faster route towards PacMan and isnt a wall (makes ghost run away)

        else: #If PacMan can get eaten
            if random.uniform(0,1) <= chase_prob: #Do best move with a chance of chase_prob
                return self.bfsPathFinding(ghost_pos) #Move with shortest path towards PacMan
            else:
                valid_moves = []

                for direction_x, direction_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  #up, down, left, right
                    next_x, next_y = x + direction_x, y + direction_y
                    if self.grid[next_x, next_y] != 1: #not wall
                        valid_moves.append((next_x, next_y))

                return random.choice(valid_moves)

    #Respawn a ghost with a Euclidian distance of 10 away from PacMan
    def respawnGhost(self, ghost_index):
        distance = 10

        empty_cells = list(zip(*np.where((self.grid == 0) | (self.grid == 4) | (self.grid == 2)))) #all cells that arent a wall or PacMan
        
        valid_cells = [cell for cell in empty_cells if (math.dist(self.pacman_pos, self.ghosts[ghost_index]) <= distance)] #all squares that are far enough from PacMan

        if(valid_cells == None):
            print("No cell found to respawn ghost")
        else:
            new_ghost_x, new_ghost_y = random.choice(valid_cells) #random square at that is atleast 10 away from PacMan
            self.ghosts[ghost_index] = (new_ghost_x, new_ghost_y) # set ghost position

    #Do a step in game loop
    def step(self, use_greedy_strategy):
        
        self.updateRMState() #update the reward machine state

        #store current PacMan Position
        current_PacMan_pos = self.pacman_pos
        #store current state before any movements take place
        state = self.getState()

        #store chosen action for pacman
        new_pos_pacMan, action = self.chooseActionPacMan(use_greedy_strategy)#choose action

        #store old ghosts positions of current state
        old_ghost_positions = []
        for i in range(self.ghost_count):
            ghost_x, ghost_y = self.ghosts[i]
            old_ghost_positions.append((ghost_x, ghost_y))

        #do ghosts actions
        for i in range(self.ghost_count):
            ghost_x, ghost_y = self.ghosts[i]
            new_pos = self.ghostSemiRandomMove((ghost_x, ghost_y), 0.7)
            self.ghosts[i] = (new_pos[0], new_pos[1])

        #do the action that was chosen by pacman before ghosts moved & store reward
        reward = self.movePacMan(current_PacMan_pos, new_pos_pacMan, old_ghost_positions)

        if(not use_greedy_strategy):#only update q_table if training
           self.updateQTable(state = state, action=action, reward=reward, next_state=self.getState())
        
        #if pacman is empowered he takes an extra action
        if(self.pill_active):
            #store current PacMan Position
            current_PacMan_pos = self.pacman_pos
            state = self.getState()

            #chosen action by PacMan
            new_pos_pacMan, action = self.chooseActionPacMan(use_greedy_strategy)#choose extra action

            #do chosen action by PacMan & store reward
            reward = self.movePacMan(current_PacMan_pos, new_pos_pacMan, old_ghost_positions)#do extra action

            if(not use_greedy_strategy):#only update q_table if training
                self.updateQTable(state = state, action=action, reward=reward, next_state=self.getState())

            #reduce pill duration
            self.pill_duration -= 1
            if(self.pill_duration <= 0): #deactivate pill if duration 0
                self.pill_active = False
            self.updateRMState() #update the reward machine state

        if(self.lives <= 0):
            self.play = False
            self.won = 0
            #print("Out of lives")
            #self.reset()
        elif(not self.pill_active and len(self.pills) <= 0): #if all pills are gone/picked up and PacMan doesnt have a pill active
            self.play = False #end game

    #reset the environment
    def reset(self):
        self.grid = self.loadMaze()  # Reload maze
        self.lives = 1  # Reset lives
        self.score = 0  # Reset score
        self.won = 0
        self.ghosts_eaten = 0
        self.pill_count = self.start_pill_count #amount of pills in the maze
        self.ghosts = [] #ghost positions
        self.pills = [] #pill positions
        self.food = [] # food positions
        self.pacman_pos = () #pacman position
        self.pill_duration = 0 #amount of moves left before pill goes unactive
        self.pill_active = False
        self.fillMaze()  # Fill maze again with Pills, Ghosts, Food & PacMan
        self.food_count = np.count_nonzero(self.grid == 4)  # Amount of food
        self.updateRMState()
        
    #shows the state of the game
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