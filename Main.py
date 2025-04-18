import gymnasium as gym
from matplotlib import pyplot as plt
from CustomPacMan import CustomPacManEnv
import time
import pickle
import numpy as np
import random
import os
import sys
from GhostHunter import GhostHunterEnv

def main():

    #Run normal version of pacman
    NormalPacMan("small")
    
    #Run ghost hunter version of pacman
    ghostHunter("small")

#
def NormalPacMan(maze_size):
    env = CustomPacManEnv(maze_size=maze_size, algo="QRM") #create environment using maze size and QRM algorithm

    min_epsilon = 0.01 #minumum epsilon value
    decay_rate = 1.33e-06 # how fast epsilon decays

    start_episode = 0 #what episode the loop starts on
    number_of_training_games = 1000000 #how many training games to perform
    test_every_X_games = 10000 #how often training is done
    number_of_testing_games = 1000 #how many games are done during training

    q_table_size = [] #stores the length of the q-table for each training episode (used for graphs)
    eps = [] #stores the epsilon value for each training episode (used for graphs)
    training_scores = [] #stores the training scores for each training episode (used for graphs)
    training_steps_taken = [] #stores the number of steps taken for each training episode (used for graphs)
    training_win = [] #stores if the game was won for each training episode (used for graphs) using 1 or 0
    training_ghosts_eaten = [] #stores the number of ghosts eaten for each training episode (used for graphs)

    testing_scores = [] #stores the average testing scores for each time testing is done (used for graphs)
    testing_steps_taken = [] #stores the average steps taken for each time testing is done (used for graphs)
    testing_win = [] #stores the average winrate for each time testing is done (used for graphs)
    testing_ghosts_eaten = [] #stores the average number of ghosts eaten for each time testing is done (used for graphs)

    testing_episode_x = [] #stores the number of training episodes that was done each time testing is done (used for graphs)

    #load in checkpoint if exists
    if os.path.exists("normal_training_QRM_checkpoint.pkl"):
        with open("normal_training_QRM_checkpoint.pkl", "rb") as f:
            saved_data = pickle.load(f)
        start_episode = saved_data['episode'] + 1
        env.q_table = saved_data['q_table']
        q_table_size = saved_data['q_table_size']
        env.epsilon = saved_data['epsilon']
        eps = saved_data['eps']
        training_scores = saved_data['training_scores']
        training_steps_taken = saved_data['training_steps_taken']
        training_win = saved_data['training_win']
        training_ghosts_eaten = saved_data['training_ghosts_eaten']
        testing_scores = saved_data['testing_scores']
        testing_steps_taken = saved_data['testing_steps_taken']
        testing_win = saved_data['testing_win']
        testing_ghosts_eaten = saved_data['testing_ghosts_eaten']
        testing_episode_x = saved_data['testing_episode_x']
        print("Loaded checkpoint from episode", saved_data['episode'])

    #training loop
    for i in range(start_episode, number_of_training_games):
        
        steps_taken = 0 #steps taken during loop

        #reduce epsilon
        if(i <= 500000):
            env.epsilon = max(min_epsilon, env.epsilon - decay_rate)
        else: #after 500000 episodes reduce epsilon by half of decay rate
            env.epsilon = max(min_epsilon, env.epsilon - (decay_rate/2))

        env.play = True

        #remove random food to help learn states
        num_food_to_remove = random.randint(0, env.food_count-1)#pick number of food pellets to remove (0 to all but 1 food)
        foods_to_remove = random.sample(env.food, num_food_to_remove)# pick random food pellets to remove
        for food_pos in foods_to_remove:
            env.food.remove(food_pos)
            env.grid[food_pos] = 0  # Clear food from grid
            env.food_count -= 1

        #start at meaning full points of game to help learning. for example start on pill 1 position
        if(i % 4 == 1):#start training episode with first pill already picked up
            env.pill_active = True
            env.pill_duration = env.pill_start_duration
            env.pill_count -= 1
            pill_pos = env.pills[0]
            env.pills.remove(pill_pos)#remove pill from list
            env.grid[pill_pos] = 0 #Clear field
            env.pacman_pos = pill_pos
            env.grid[pill_pos] = 5 # set grid square to PacMan

        elif(i % 4 == 2):#start training episode with second pill already picked up
            env.pill_active = True
            env.pill_duration = env.pill_start_duration
            env.pill_count -= 1
            pill_pos = env.pills[1]
            env.pills.remove(pill_pos)
            env.grid[pill_pos] = 0 #Clear field
            env.pacman_pos = pill_pos
            env.grid[pill_pos] = 5 # set grid square to PacMan

        elif(i % 4 == 3):#start training episode with both pills already picked up
            env.pill_active = True
            env.pill_duration = env.pill_start_duration
            env.pill_count -= 2
            pill_pos1 = env.pills[0]
            env.pills.remove(pill_pos1)
            env.grid[pill_pos1] = 0 #Clear field
            pill_pos2 = env.pills[0]
            env.pills.remove(pill_pos2)
            env.grid[pill_pos2] = 0 #Clear field

            pill_pos = random.choice([pill_pos1, pill_pos2])#pick random pill position
            env.pacman_pos = pill_pos #set pacmans position to pill position
            env.grid[pill_pos] = 5 # set grid square to PacMan


        while(env.play == True): # game loop
            env.step(use_greedy_strategy=False)#do game step
            steps_taken += 1
            if(steps_taken >= 1000):
                print("training game took to long")
                env.play = False # stop game if >= 1000 steps taken
                
        #append measurements to lists when game finished
        eps.append(env.epsilon)
        q_table_size.append(len(env.q_table))
        training_scores.append(env.score)
        training_steps_taken.append(steps_taken)
        training_win.append(env.won)
        training_ghosts_eaten.append(env.ghosts_eaten)

        #reset environment
        env.reset()

        #test every X number of episodes
        if(i % test_every_X_games == 0):
                print(i)
                print("q-table size: ", len(env.q_table))
                #print("ëpsilon:", env.epsilon)
                testing_episode_x.append(i)
                testing(number_of_testing_games, env, testing_scores, testing_steps_taken, testing_win, testing_ghosts_eaten) #start testing loop

        #save checkpoint every 250k episodes
        if(i % 250000 == 0 and i != 0):
            save_data = {
                'episode': i,
                'q_table': env.q_table,
                'q_table_size': q_table_size,
                'epsilon': env.epsilon,
                'eps': eps,
                'training_scores': training_scores,
                'training_steps_taken': training_steps_taken,
                'training_win': training_win,
                'training_ghosts_eaten': training_ghosts_eaten,
                'testing_scores': testing_scores,
                'testing_steps_taken': testing_steps_taken,
                'testing_win': testing_win,
                'testing_ghosts_eaten': testing_ghosts_eaten,
                'testing_episode_x': testing_episode_x
            }
            with open("normal_training_QRM_checkpoint.pkl", "wb") as f:
                pickle.dump(save_data, f)
            print("Saved checkpoint")

    #close environment
    env.close()

#---------------------------QL algo-----------------------------------------

    env = CustomPacManEnv(maze_size=maze_size, algo="QL") #create environment using maze size and QL algorithm

    min_epsilon = 0.01 #minumum epsilon value
    decay_rate = 1.33e-06 # how fast epsilon decays

    start_episode = 0 #what episode the loop starts on
    number_of_training_games = 1000000 #how many training games to perform
    test_every_X_games = 10000 #how often training is done
    number_of_testing_games = 1000 #how many games are done during training

    #lists for QL agent measurements
    QL_q_table_size = []
    QL_eps = []
    QL_training_scores = []
    QL_training_steps_taken = []#steps until game ends
    QL_training_win = []#game won or lost (1 or 0)
    QL_training_ghosts_eaten = []

    QL_testing_scores = []
    QL_testing_steps_taken = []#steps until game ends
    QL_testing_win = []#game won or lost (1 or 0)
    QL_testing_ghosts_eaten = []

    QL_testing_episode_x = []

    #load checkpoint if exists
    if os.path.exists("normal_training_QL_checkpoint.pkl"):
        with open("normal_training_QL_checkpoint.pkl", "rb") as f:
            saved_data = pickle.load(f)
        start_episode = saved_data['episode'] + 1
        env.q_table = saved_data['q_table']
        env.epsilon = saved_data['epsilon']
        QL_q_table_size = saved_data['q_table_size']
        QL_eps = saved_data['eps']
        QL_training_scores = saved_data['training_scores']
        QL_training_steps_taken = saved_data['training_steps_taken']
        QL_training_win = saved_data['training_win']
        QL_training_ghosts_eaten = saved_data['training_ghosts_eaten']
        QL_testing_scores = saved_data['testing_scores']
        QL_testing_steps_taken = saved_data['testing_steps_taken']
        QL_testing_win = saved_data['testing_win']
        QL_testing_ghosts_eaten = saved_data['testing_ghosts_eaten']
        QL_testing_episode_x = saved_data['testing_episode_x']
        print("Loaded checkpoint from episode", saved_data['episode'])

    #training loop
    for i in range(start_episode, number_of_training_games):
        
        steps_taken = 0

        #reduce epsilon
        if(i <= 500000):
            env.epsilon = max(min_epsilon, env.epsilon - decay_rate)
        else: 
            env.epsilon = max(min_epsilon, env.epsilon - (decay_rate/2))

        env.play = True

        #remove random food to help learn states
        num_food_to_remove = random.randint(0, env.food_count-1)#pick 0 to all but 1 food to remove
        foods_to_remove = random.sample(env.food, num_food_to_remove)
        for food_pos in foods_to_remove:
            env.food.remove(food_pos)
            env.grid[food_pos] = 0  # Clear food from grid
            env.food_count -= 1

        #start at meaning full points of game to help learning. for example start on pill 1 position
        if(i % 4 == 1):#start training episode with first pill already picked up
            env.pill_active = True
            env.pill_duration = env.pill_start_duration
            env.pill_count -= 1
            pill_pos = env.pills[0]
            env.pills.remove(pill_pos)
            env.grid[pill_pos] = 0 #Clear field
            env.pacman_pos = pill_pos
            env.grid[pill_pos] = 5 # set grid square to PacMan

        elif(i % 4 == 2):#start training episode with second pill already picked up
            env.pill_active = True
            env.pill_duration = env.pill_start_duration
            env.pill_count -= 1
            pill_pos = env.pills[1]
            env.pills.remove(pill_pos)
            env.grid[pill_pos] = 0 #Clear field
            env.pacman_pos = pill_pos
            env.grid[pill_pos] = 5 # set grid square to PacMan

        elif(i % 4 == 3):#start training episode with both pills already picked up
            env.pill_active = True
            env.pill_duration = env.pill_start_duration
            env.pill_count -= 2
            pill_pos1 = env.pills[0]
            env.pills.remove(pill_pos1)
            env.grid[pill_pos1] = 0 #Clear field
            pill_pos2 = env.pills[0]
            env.pills.remove(pill_pos2)
            env.grid[pill_pos2] = 0 #Clear field

            pill_pos = random.choice([pill_pos1, pill_pos2])#pick random pill position
            env.pacman_pos = pill_pos #set pacmans position to pill position
            env.grid[pill_pos] = 5 # set grid square to PacMan


        while(env.play == True): # game loop
            env.step(use_greedy_strategy=False)
            steps_taken += 1
            if(steps_taken >= 1000):
                print("training game took to long")
                env.play = False
        
        #append measurements to lists after game ends
        QL_eps.append(env.epsilon)
        QL_q_table_size.append(len(env.q_table))
        QL_training_scores.append(env.score)
        QL_training_steps_taken.append(steps_taken)
        QL_training_win.append(env.won)
        QL_training_ghosts_eaten.append(env.ghosts_eaten)

        #reset environment
        env.reset()

        #test every X number of episodes
        if(i % test_every_X_games == 0):
                print(i)
                print("q-table size: ", len(env.q_table))
                #print("ëpsilon:", env.epsilon)
                QL_testing_episode_x.append(i)
                testing(number_of_testing_games, env, QL_testing_scores, QL_testing_steps_taken, QL_testing_win, QL_testing_ghosts_eaten) #start testing loop

        #create checkpoint every 250000 episodes
        if(i % 250000 == 0 and i != 0):
            save_data = {
                'episode': i,
                'q_table': env.q_table,
                'q_table_size': QL_q_table_size,
                'epsilon': env.epsilon,
                'eps': QL_eps,
                'training_scores': QL_training_scores,
                'training_steps_taken': QL_training_steps_taken,
                'training_win': QL_training_win,
                'training_ghosts_eaten': QL_training_ghosts_eaten,
                'testing_scores': QL_testing_scores,
                'testing_steps_taken': QL_testing_steps_taken,
                'testing_win': QL_testing_win,
                'testing_ghosts_eaten': QL_testing_ghosts_eaten,
                'testing_episode_x': QL_testing_episode_x
            }
            with open("normal_training_QL_checkpoint.pkl", "wb") as f:
                pickle.dump(save_data, f)
            print("Saved checkpoint")

    #---------------------------Create Graphs-----------------------------------------------

    #------------------create training episode graphs------------------------

    #q-table graph
    window = 10000 #window for moving average
    training_q_table_size_normal_qrm_agent_1_moving_avg = np.convolve(q_table_size, np.ones(window)/window, mode='valid') #moving average QRM
    training_q_table_size_normal_ql_agent_1_moving_avg = np.convolve(QL_q_table_size, np.ones(window)/window, mode='valid') #moving average QL
    plt.plot(training_q_table_size_normal_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(training_q_table_size_normal_ql_agent_1_moving_avg, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("q_table size")
    plt.title("Q-table size over training episodes (small map)")
    plt.grid(True)
    plt.savefig("q-table-size-QRM-small-1.png", dpi=300, bbox_inches='tight')
    plt.show()

    #epsilon graph
    plt.plot(eps, color = 'black', label='epsilon value')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("epsilon")
    plt.title("epsilon over training episodes (small map)")
    plt.grid(True)
    plt.savefig("epsilon-QRM-small-1.png", dpi=300, bbox_inches='tight')
    plt.show()

    #training scores graph
    window = 10000
    training_scores_normal_qrm_agent_1_moving_avg = np.convolve(training_scores, np.ones(window)/window, mode='valid')
    training_scores_normal_ql_agent_1_moving_avg = np.convolve(QL_training_scores, np.ones(window)/window, mode='valid')
    plt.plot(training_scores_normal_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(training_scores_normal_ql_agent_1_moving_avg, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Score per training episode (small map)")
    plt.grid(True)
    plt.savefig("training-scores-QRM-small-1.png", dpi=300, bbox_inches='tight')
    plt.show()

    #number of steps taken during training episodes graph
    window = 10000
    training_steps_normal_qrm_agent_1_moving_avg = np.convolve(training_steps_taken, np.ones(window)/window, mode='valid')
    training_steps_normal_ql_agent_1_moving_avg = np.convolve(QL_training_steps_taken, np.ones(window)/window, mode='valid')
    plt.plot(training_steps_normal_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(training_steps_normal_ql_agent_1_moving_avg, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()    
    plt.xlabel("Episode")
    plt.ylabel("Steps taken")
    plt.title("Steps taken per training episode (small map)")
    plt.grid(True)
    plt.savefig("training-steps-QRM-small-1", dpi=300, bbox_inches='tight')
    plt.show()

    #winrate during training episodes graph
    window = 10000
    training_win_normal_qrm_agent_1_moving_avg = np.convolve(training_win, np.ones(window)/window, mode='valid')
    training_win_normal_ql_agent_1_moving_avg = np.convolve(QL_training_win, np.ones(window)/window, mode='valid')
    plt.plot(training_win_normal_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(training_win_normal_ql_agent_1_moving_avg, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Winrate")
    plt.title("Winrate per training episode (small map)")
    plt.grid(True)
    plt.savefig("training-win-QRM-small-1", dpi=300, bbox_inches='tight')
    plt.show()

    #number of ghosts eaten during training graph
    window = 10000
    training_eaten_normal_qrm_agent_1_moving_avg = np.convolve(training_ghosts_eaten, np.ones(window)/window, mode='valid')
    training_eaten_normal_ql_agent_1_moving_avg = np.convolve(QL_training_ghosts_eaten, np.ones(window)/window, mode='valid')
    plt.plot(training_eaten_normal_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(training_eaten_normal_ql_agent_1_moving_avg, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Number of ghosts eaten")
    plt.title("Number of ghosts eaten per training episode (small map)")
    plt.grid(True)
    plt.savefig("training-eaten-QRM-small-1", dpi=300, bbox_inches='tight')
    plt.show()

    #-------------------------graphs for test episodes-------------------------#

    #test scores graph
    plt.plot(testing_episode_x, testing_scores, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(QL_testing_episode_x, QL_testing_scores, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Average score")
    plt.title("Average test score every 10k training episodes (small map)")
    plt.grid(True)
    plt.savefig("test-scores-QRM-small-1", dpi=300, bbox_inches='tight')
    plt.show()

    #number of steps taken during testing graph
    plt.plot(testing_episode_x, testing_steps_taken, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(QL_testing_episode_x, QL_testing_steps_taken, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Average number of steps taken")
    plt.title("Average number of steps taken every 10k training episodes (small map)")
    plt.grid(True)
    plt.savefig("test-steps-QRM-small-1", dpi=300, bbox_inches='tight')
    plt.show()

    #winrate during testing graph
    plt.plot(testing_episode_x, testing_win, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(QL_testing_episode_x, QL_testing_win, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Average Winrate")
    plt.title("Average winrate every 10k training episodes (small map)")
    plt.grid(True)
    plt.savefig("test-win-QRM-small-1", dpi=300, bbox_inches='tight')
    plt.show()

    #number of ghosts eaten during testing graph
    plt.plot(testing_episode_x, testing_ghosts_eaten, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(QL_testing_episode_x, QL_testing_ghosts_eaten, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Average Number of ghosts eaten")
    plt.title("Average Number of ghosts eaten every 10k training episodes (small map)")
    plt.grid(True)
    plt.savefig("test-eaten-QRM-small-1", dpi=300, bbox_inches='tight')
    plt.show()

    #---------see for your self--------------
    for i in range(5):

        env.play = True
        while(env.play == True):
            env.step(use_greedy_strategy=True)
            env.render() #render game
            time.sleep(0.25) #wait 0.25 seconds to see game

        env.reset()
    #-----------------------------------------

    env.close()#close environment

#test loop
def testing(number_of_testing_games, env, testing_scores, testing_steps_taken, testing_win, testing_ghosts_eaten):
        score = 0
        steps = 0
        wins = 0
        ghosts_eaten = 0
        
        for i in range(number_of_testing_games):

            steps_taken = 0

            env.play = True
            while(env.play == True):
                env.step(use_greedy_strategy=True)#use greedy during testing
                steps_taken += 1
                if(steps_taken >= 1000):
                    #print("game took to long")
                    env.play = False

            #append scores
            score += env.score
            steps += steps_taken
            wins += env.won
            ghosts_eaten += env.ghosts_eaten

            env.reset()

        #calculate average of measurements
        avg_score = score / number_of_testing_games
        avg_steps = steps / number_of_testing_games
        avg_win = wins / number_of_testing_games
        avg_ghosts_eaten = ghosts_eaten / number_of_testing_games

        #append average of measurements to lists
        testing_scores.append(avg_score)
        testing_steps_taken.append(avg_steps)
        testing_win.append(avg_win)
        testing_ghosts_eaten.append(avg_ghosts_eaten)
        print("avg score testing: ", avg_score)
        print("avg steps: ", avg_steps)
        print("avg ghosts eaten: ", avg_ghosts_eaten)



#train and test both algorithms on the ghosthunter environment
def ghostHunter(maze_size):
    env = GhostHunterEnv(maze_size=maze_size, algo="QRM")

    min_epsilon = 0.01 #minumum epsilon value
    decay_rate = 1.33e-06 # how fast epsilon decays

    start_episode = 0 #what episode the loop starts on
    number_of_training_games = 1000000 #how many training games to perform
    test_every_X_games = 10000 #how often training is done
    number_of_testing_games = 1000 #how many games are done during training

    #store measurements for each training episode
    q_table_size = []
    eps = []
    training_scores = []
    training_steps_taken = []#steps until game ends
    training_win = []#game won or lost (1 or 0)
    training_ghosts_eaten = []

    testing_scores = []
    testing_steps_taken = []#steps until game ends
    testing_win = []#game won or lost (1 or 0)
    testing_ghosts_eaten = []

    testing_episode_x = []

    #load checkpoint if exists
    if os.path.exists("ghost_training_QRM_checkpoint.pkl"):
        with open("ghost_training_QRM_checkpoint.pkl", "rb") as f:
            saved_data = pickle.load(f)
        start_episode = saved_data['episode'] + 1
        env.q_table = saved_data['q_table']
        q_table_size = saved_data['q_table_size']
        env.epsilon = saved_data['epsilon']
        eps = saved_data['eps']
        training_scores = saved_data['training_scores']
        training_steps_taken = saved_data['training_steps_taken']
        training_win = saved_data['training_win']
        training_ghosts_eaten = saved_data['training_ghosts_eaten']
        testing_scores = saved_data['testing_scores']
        testing_steps_taken = saved_data['testing_steps_taken']
        testing_win = saved_data['testing_win']
        testing_ghosts_eaten = saved_data['testing_ghosts_eaten']
        testing_episode_x = saved_data['testing_episode_x']
        print("Loaded checkpoint from episode", saved_data['episode'])

    #training loop
    for i in range(start_episode, number_of_training_games):
        
        steps_taken = 0

        #decay epsilon
        if(i <= 500000):
            env.epsilon = max(min_epsilon, env.epsilon - decay_rate)
        else: 
            env.epsilon = max(min_epsilon, env.epsilon - (decay_rate/2))

        env.play = True

        #start at meaning full points of game to help learning. for example start on pill 1 position
        if(i % 4 == 1):#start training episode with first pill already picked up
            env.pill_active = True
            env.pill_duration = env.pill_start_duration
            env.pill_count -= 1
            pill_pos = env.pills[0]
            env.pills.remove(pill_pos)
            env.grid[pill_pos] = 0 #Clear field
            env.pacman_pos = pill_pos
            env.grid[pill_pos] = 5 # set grid square to PacMan

        elif(i % 4 == 2):#start training episode with second pill already picked up
            env.pill_active = True
            env.pill_duration = env.pill_start_duration
            env.pill_count -= 1
            pill_pos = env.pills[1]
            env.pills.remove(pill_pos)
            env.grid[pill_pos] = 0 #Clear field
            env.pacman_pos = pill_pos
            env.grid[pill_pos] = 5 # set grid square to PacMan

        elif(i % 4 == 3):#start training episode with both pills already picked up
            env.pill_active = True
            env.pill_duration = env.pill_start_duration
            env.pill_count -= 2
            pill_pos1 = env.pills[0]
            env.pills.remove(pill_pos1)
            env.grid[pill_pos1] = 0 #Clear field
            pill_pos2 = env.pills[0]
            env.pills.remove(pill_pos2)
            env.grid[pill_pos2] = 0 #Clear field

            pill_pos = random.choice([pill_pos1, pill_pos2])#pick random pill position
            env.pacman_pos = pill_pos #set pacmans position to pill position
            env.grid[pill_pos] = 5 # set grid square to PacMan


        while(env.play == True): # game loop
            env.step(use_greedy_strategy=False)
            steps_taken += 1
            if(steps_taken >= 1000):
                print("training game took to long")
                env.play = False
        
        #append measurement for training episode after game ends
        eps.append(env.epsilon)
        q_table_size.append(len(env.q_table))
        training_scores.append(env.score)
        training_steps_taken.append(steps_taken)
        training_win.append(env.won)
        training_ghosts_eaten.append(env.ghosts_eaten)

        #reset environment   
        env.reset()

        #test after every X training episodes
        if(i % test_every_X_games == 0):
                print(i)
                print("q-table size: ", len(env.q_table))
                #print("ëpsilon:", env.epsilon)
                testing_episode_x.append(i)
                testing(number_of_testing_games, env, testing_scores, testing_steps_taken, testing_win, testing_ghosts_eaten) #run test loop

        #save checkpoint every 250000 episodes       
        if(i % 250000 == 0 and i != 0):
            save_data = {
                'episode': i,
                'q_table': env.q_table,
                'q_table_size': q_table_size,
                'epsilon': env.epsilon,
                'eps': eps,
                'training_scores': training_scores,
                'training_steps_taken': training_steps_taken,
                'training_win': training_win,
                'training_ghosts_eaten': training_ghosts_eaten,
                'testing_scores': testing_scores,
                'testing_steps_taken': testing_steps_taken,
                'testing_win': testing_win,
                'testing_ghosts_eaten': testing_ghosts_eaten,
                'testing_episode_x': testing_episode_x
            }
            with open("ghost_training_QRM_checkpoint.pkl", "wb") as f:
                pickle.dump(save_data, f)
            print("Saved checkpoint")

    #close environment
    env.close()
    
    #------------------QL agent-----------------------#

    env = GhostHunterEnv(maze_size=maze_size, algo="QL") #create environment with QL algo

    min_epsilon = 0.01 #minumum epsilon value
    decay_rate = 1.33e-06 # how fast epsilon decays

    start_episode = 0 #what episode the loop starts on
    number_of_training_games = 1000000 #how many training games to perform
    test_every_X_games = 10000 #how often training is done
    number_of_testing_games = 1000 #how many games are done during training

    #store measurements for each training game of QL agent
    QL_q_table_size = []
    QL_eps = []
    QL_training_scores = []
    QL_training_steps_taken = []#steps until game ends
    QL_training_win = []#game won or lost (1 or 0)
    QL_training_ghosts_eaten = []

    QL_testing_scores = []
    QL_testing_steps_taken = []#steps until game ends
    QL_testing_win = []#game won or lost (1 or 0)
    QL_testing_ghosts_eaten = []

    QL_testing_episode_x = []

    #load checkpoint if exists
    if os.path.exists("ghost_training_QL_checkpoint.pkl"):
        with open("ghost_training_QL_checkpoint.pkl", "rb") as f:
            saved_data = pickle.load(f)
        start_episode = saved_data['episode'] + 1
        env.q_table = saved_data['q_table']
        env.epsilon = saved_data['epsilon']
        QL_q_table_size = saved_data['q_table_size']
        QL_eps = saved_data['eps']
        QL_training_scores = saved_data['training_scores']
        QL_training_steps_taken = saved_data['training_steps_taken']
        QL_training_win = saved_data['training_win']
        QL_training_ghosts_eaten = saved_data['training_ghosts_eaten']
        QL_testing_scores = saved_data['testing_scores']
        QL_testing_steps_taken = saved_data['testing_steps_taken']
        QL_testing_win = saved_data['testing_win']
        QL_testing_ghosts_eaten = saved_data['testing_ghosts_eaten']
        QL_testing_episode_x = saved_data['testing_episode_x']
        print("Loaded checkpoint from episode", saved_data['episode'])

    #training loop
    for i in range(start_episode, number_of_training_games):
        
        steps_taken = 0

        #decay epsilon
        if(i <= 500000):
            env.epsilon = max(min_epsilon, env.epsilon - decay_rate)
        else: 
            env.epsilon = max(min_epsilon, env.epsilon - (decay_rate/2))

        env.play = True

        #start at meaning full points of game to help learning. for example start on pill 1 position
        if(i % 4 == 1):#start training episode with first pill already picked up
            env.pill_active = True
            env.pill_duration = env.pill_start_duration
            env.pill_count -= 1
            pill_pos = env.pills[0]
            env.pills.remove(pill_pos)
            env.grid[pill_pos] = 0 #Clear field
            env.pacman_pos = pill_pos
            env.grid[pill_pos] = 5 # set grid square to PacMan

        elif(i % 4 == 2):#start training episode with second pill already picked up
            env.pill_active = True
            env.pill_duration = env.pill_start_duration
            env.pill_count -= 1
            pill_pos = env.pills[1]
            env.pills.remove(pill_pos)
            env.grid[pill_pos] = 0 #Clear field
            env.pacman_pos = pill_pos
            env.grid[pill_pos] = 5 # set grid square to PacMan

        elif(i % 4 == 3):#start training episode with both pills already picked up
            env.pill_active = True
            env.pill_duration = env.pill_start_duration
            env.pill_count -= 2
            pill_pos1 = env.pills[0]
            env.pills.remove(pill_pos1)
            env.grid[pill_pos1] = 0 #Clear field
            pill_pos2 = env.pills[0]
            env.pills.remove(pill_pos2)
            env.grid[pill_pos2] = 0 #Clear field

            pill_pos = random.choice([pill_pos1, pill_pos2])#pick random pill position
            env.pacman_pos = pill_pos #set pacmans position to pill position
            env.grid[pill_pos] = 5 # set grid square to PacMan


        while(env.play == True): # game loop
            env.step(use_greedy_strategy=False)
            steps_taken += 1
            if(steps_taken >= 1000):
                print("training game took to long")
                env.play = False
                
        #append measurements of training episode after each game
        QL_eps.append(env.epsilon)
        QL_q_table_size.append(len(env.q_table))
        QL_training_scores.append(env.score)
        QL_training_steps_taken.append(steps_taken)
        QL_training_win.append(env.won)
        QL_training_ghosts_eaten.append(env.ghosts_eaten)

        #reset environment
        env.reset()

        #test after every X number of training games
        if(i % test_every_X_games == 0):
                print(i)
                print("q-table size: ", len(env.q_table))
                #print("ëpsilon:", env.epsilon)
                QL_testing_episode_x.append(i)
                testing(number_of_testing_games, env, QL_testing_scores, QL_testing_steps_taken, QL_testing_win, QL_testing_ghosts_eaten)# start test loop
    
        #save checkpoint every 250000 episodes
        if(i % 250000 == 0 and i != 0):
            save_data = {
                'episode': i,
                'q_table': env.q_table,
                'q_table_size': q_table_size,
                'epsilon': env.epsilon,
                'eps': QL_eps,
                'training_scores': QL_training_scores,
                'training_steps_taken': QL_training_steps_taken,
                'training_win': QL_training_win,
                'training_ghosts_eaten': QL_training_ghosts_eaten,
                'testing_scores': QL_testing_scores,
                'testing_steps_taken': QL_testing_steps_taken,
                'testing_win': QL_testing_win,
                'testing_ghosts_eaten': QL_testing_ghosts_eaten,
                'testing_episode_x': QL_testing_episode_x
            }
            with open("ghost_training_QL_checkpoint.pkl", "wb") as f:
                pickle.dump(save_data, f)
            print("Saved checkpoint")


    #-----------------------------------create graphs------------------------#

    #-------------------------create graphs for training games-------------------------

    #q-table graph
    window = 10000 #moving average window
    training_q_table_size_normal_qrm_agent_1_moving_avg = np.convolve(q_table_size, np.ones(window)/window, mode='valid') #moving average QRM agent
    training_q_table_size_normal_ql_agent_1_moving_avg = np.convolve(QL_q_table_size, np.ones(window)/window, mode='valid') #moving average QL agent
    plt.plot(training_q_table_size_normal_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(training_q_table_size_normal_ql_agent_1_moving_avg, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("q_table size")
    plt.title("Q-table size over training episodes (ghost hunter small map)")
    plt.grid(True)
    plt.savefig("q-table-size-QRM-ghost-hunter-small.png", dpi=300, bbox_inches='tight')
    plt.show()

    #epsilon graph
    plt.plot(eps, color = 'black', label='epsilon value')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("epsilon")
    plt.title("epsilon over training episodes (ghost hunter small map)")
    plt.grid(True)
    plt.savefig("epsilon-QRM-ghost-hunter-small.png", dpi=300, bbox_inches='tight')
    plt.show()

    #training scores graph
    window = 10000
    training_scores_normal_qrm_agent_1_moving_avg = np.convolve(training_scores, np.ones(window)/window, mode='valid')
    training_scores_normal_ql_agent_1_moving_avg = np.convolve(QL_training_scores, np.ones(window)/window, mode='valid')
    plt.plot(training_scores_normal_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(training_scores_normal_ql_agent_1_moving_avg, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Score per training episode (ghost hunter small map)")
    plt.grid(True)
    plt.savefig("training-scores-QRM-ghost-hunter-small.png", dpi=300, bbox_inches='tight')
    plt.show()

    #steps taken
    window = 10000
    training_steps_normal_qrm_agent_1_moving_avg = np.convolve(training_steps_taken, np.ones(window)/window, mode='valid')
    training_steps_normal_ql_agent_1_moving_avg = np.convolve(QL_training_steps_taken, np.ones(window)/window, mode='valid')
    plt.plot(training_steps_normal_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(training_steps_normal_ql_agent_1_moving_avg, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()    
    plt.xlabel("Episode")
    plt.ylabel("Steps taken")
    plt.title("Steps taken per training episode (ghost hunter small map)")
    plt.grid(True)
    plt.savefig("training-steps-QRM-ghost-hunter-small", dpi=300, bbox_inches='tight')
    plt.show()

    #winrate
    window = 10000
    training_win_normal_qrm_agent_1_moving_avg = np.convolve(training_win, np.ones(window)/window, mode='valid')
    training_win_normal_ql_agent_1_moving_avg = np.convolve(QL_training_win, np.ones(window)/window, mode='valid')
    plt.plot(training_win_normal_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(training_win_normal_ql_agent_1_moving_avg, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Winrate")
    plt.title("Winrate per training episode (ghost hunter small map)")
    plt.grid(True)
    plt.savefig("training-win-QRM-ghost-hunter-small", dpi=300, bbox_inches='tight')
    plt.show()

    #ghosts eaten
    window = 10000
    training_eaten_normal_qrm_agent_1_moving_avg = np.convolve(training_ghosts_eaten, np.ones(window)/window, mode='valid')
    training_eaten_normal_ql_agent_1_moving_avg = np.convolve(QL_training_ghosts_eaten, np.ones(window)/window, mode='valid')
    plt.plot(training_eaten_normal_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(training_eaten_normal_ql_agent_1_moving_avg, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Number of ghosts eaten")
    plt.title("Number of ghosts eaten per training episode (ghost hunter small map)")
    plt.grid(True)
    plt.savefig("training-eaten-QRM-ghost-hunter-small", dpi=300, bbox_inches='tight')
    plt.show()

    #-------------------------graphs for testing games-------------------------#

    #scores
    plt.plot(testing_episode_x, testing_scores, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(QL_testing_episode_x, QL_testing_scores, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Average score")
    plt.title("Average test score every 10k training episodes (ghost hunter small map)")
    plt.grid(True)
    plt.savefig("test-scores-QRM-ghost-hunter-small", dpi=300, bbox_inches='tight')
    plt.show()

    #steps
    plt.plot(testing_episode_x, testing_steps_taken, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(QL_testing_episode_x, QL_testing_steps_taken, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Average number of steps taken")
    plt.title("Average number of steps taken every 10k training episodes (ghost hunter small map)")
    plt.grid(True)
    plt.savefig("test-steps-QRM-ghost-hunter-small", dpi=300, bbox_inches='tight')
    plt.show()

    #winrate
    plt.plot(testing_episode_x, testing_win, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(QL_testing_episode_x, QL_testing_win, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Average Winrate")
    plt.title("Average winrate every 10k training episodes (ghost hunter small map)")
    plt.grid(True)
    plt.savefig("test-win-QRM-ghost-hunter-small", dpi=300, bbox_inches='tight')
    plt.show()

    #ghosts eaten
    plt.plot(testing_episode_x, testing_ghosts_eaten, color = 'black', label='Q-Learning with RM Agent')
    plt.plot(QL_testing_episode_x, QL_testing_ghosts_eaten, color = 'grey', linestyle="--", label='Q-Learning Agent')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Average Number of ghosts eaten")
    plt.title("Average Number of ghosts eaten every 10k training episodes (ghost hunter small map)")
    plt.grid(True)
    plt.savefig("test-eaten-QRM-ghost-hunter-small", dpi=300, bbox_inches='tight')
    plt.show()

    #---------see for your self--------------
    for i in range(5):

        env.play = True
        while(env.play == True):
            env.step(use_greedy_strategy=True)
            env.render()
            time.sleep(0.25)

        env.reset()
    #-----------------------------------------

    env.close()

#save q-table
def save_q_table(self, filename="q_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(self.q_table, f)

#load q-table
def load_q_table(self, filename="q_table.pkl"):
    with open(filename, "rb") as f:
        self.q_table = pickle.load(f)

if __name__=="__main__":
    main()