import gymnasium as gym
from matplotlib import pyplot as plt
from CustomPacMan import CustomPacManEnv
import time
import pickle
import numpy as np
import random

def main():
    env = CustomPacManEnv(maze_size="small", algo="QRM")

    min_epsilon = 0.05 #0.01
    decay_rate = 1.05e-06 #5.5e-07 #2.2e-06

    number_of_training_games = 2000000 #500000
    test_every_X_games = 10000
    number_of_testing_games = 1000 #2500 #5000

    q_table_size = []
    eps = []
    training_scores = []
    training_steps_taken = []#steps until game ends
    training_win = []#game won or lost (1 or 0)

    testing_scores = []
    testing_steps_taken = []#steps until game ends
    testing_win = []#game won or lost (1 or 0)

    testing_episode_x = []


    #load_q_table(env, "small-ql.pkl")
    #print("q-table size: ", len(env.q_table))

    #print(env.grid)
    for i in range(number_of_training_games):
        
        steps_taken = 0
        env.epsilon = max(min_epsilon, env.epsilon - decay_rate)
        #env.epsilon = min_epsilon + (1.0 - min_epsilon) * np.exp(-decay_rate * i)

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
            env.grid[pill_pos] = 5 # set to PacMan

        elif(i % 4 == 2):#start training episode with second pill already picked up
            env.pill_active = True
            env.pill_duration = env.pill_start_duration
            env.pill_count -= 1
            pill_pos = env.pills[1]
            env.pills.remove(pill_pos)
            env.grid[pill_pos] = 0 #Clear field
            env.pacman_pos = pill_pos
            env.grid[pill_pos] = 5 # set to PacMan

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
            env.grid[pill_pos] = 5 # set to PacMan


        while(env.play == True):
            env.step(use_greedy_strategy=False)

            steps_taken += 1
                
        eps.append(env.epsilon)
        q_table_size.append(len(env.q_table))
        training_scores.append(env.score)
        training_steps_taken.append(steps_taken)
        training_win.append(env.won)

        env.reset()

        if(i % test_every_X_games == 0):
                print(i)
                #print("ëpsilon:", env.epsilon)
                testing_episode_x.append(i)
                testing(number_of_testing_games, env, testing_scores, testing_steps_taken, testing_win)


        #print(i)
        #print("--------------------------------------------------")
        #print(env.grid)
        #print(env.pill_active)
        #env.render()
        #time.sleep(0.25)



        #print("--------------------------------------------------")
        #print(env.grid)
        #print(env.pill_active)
            #env.render()
            #time.sleep(0.25)
    #print(env.grid)

    print("q-table size: ", len(env.q_table))

    window = 10000
    training_q_table_size_small_qrm_agent_1_moving_avg = np.convolve(q_table_size, np.ones(window)/window, mode='valid')
    plt.plot(training_q_table_size_small_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning with RM Agent')
    plt.legend(loc='lower left')
    plt.xlabel("Episode")
    plt.ylabel("q_table size")
    plt.title("Q-table size over training episodes (small map)")
    plt.savefig("q-table-size-QRM-small-15.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.plot(eps, color = 'black', label='epsilon value')
    plt.legend(loc='lower left')
    plt.xlabel("Episode")
    plt.ylabel("epsilon")
    plt.title("epsilon over training episodes (small map)")
    plt.savefig("epsilon-QRM-small-15.png", dpi=300, bbox_inches='tight')
    plt.show()

    window = 10000
    training_scores_small_qrm_agent_1_moving_avg = np.convolve(training_scores, np.ones(window)/window, mode='valid')
    plt.plot(training_scores_small_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning with RM Agent')
    plt.legend(loc='lower left')    
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Score per training episode (small map)")
    plt.savefig("training-scores-QRM-small-15.png", dpi=300, bbox_inches='tight')
    plt.show()

    window = 10000
    training_steps_small_qrm_agent_1_moving_avg = np.convolve(training_steps_taken, np.ones(window)/window, mode='valid')
    plt.plot(training_steps_small_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning with RM Agent')
    plt.legend(loc='lower left')    
    plt.xlabel("Episode")
    plt.ylabel("Steps taken")
    plt.title("Steps taken per training episode (small map)")
    plt.savefig("training-steps-QRM-small-15", dpi=300, bbox_inches='tight')
    plt.show()

    window = 10000
    training_win_small_qrm_agent_1_moving_avg = np.convolve(training_win, np.ones(window)/window, mode='valid')
    plt.plot(training_win_small_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning Agent with RM')
    plt.legend(loc='lower left')
    plt.xlabel("Episode")
    plt.ylabel("Winrate")
    plt.title("Winrate per training episode (small map)")
    plt.savefig("training-win-QRM-small-15", dpi=300, bbox_inches='tight')
    plt.show()

    #-------------------------testing-------------------------#
    #window = 5
    #test_scores_small_qrm_agent_1_moving_avg = np.convolve(testing_scores, np.ones(window)/window, mode='valid')
    #x = testing_episode_x[window - 1:]
    #print(len(x))
    #print(len(test_scores_small_qrm_agent_1_moving_avg))
    #plt.plot(x, test_scores_small_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning with RM Agent (alpla=0.05) (gamma=0.99) (epsilon=0.7)')
    plt.plot(testing_episode_x, testing_scores, color = 'black', label='Q-Learning with RM Agent')
    plt.legend(loc='lower left')
    plt.xlabel("Episode")
    plt.ylabel("Average score")
    plt.title("Average test score every 10k training episodes (small map)")
    plt.savefig("test-scores-QRM-small-15", dpi=300, bbox_inches='tight')
    plt.show()

    #window = 5
    #test_steps_small_qrm_agent_1_moving_avg = np.convolve(testing_steps_taken, np.ones(window)/window, mode='valid')
    #x = testing_episode_x[window - 1:]
    #plt.plot(x, test_steps_small_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning with RM Agent (alpla=0.05) (gamma=0.99) (epsilon=0.7)')
    plt.plot(testing_episode_x, testing_steps_taken, color = 'black', label='Q-Learning with RM Agent')
    plt.legend(loc='lower left')
    plt.xlabel("Episode")
    plt.ylabel("Average number of steps taken")
    plt.title("Average number of steps taken every 10k training episodes (small map)")
    plt.savefig("test-steps-QRM-small-15", dpi=300, bbox_inches='tight')
    plt.show()

    #window = 5
    #test_win_small_qrm_agent_1_moving_avg = np.convolve(testing_win, np.ones(window)/window, mode='valid')
    #x = testing_episode_x[window - 1:]
    #plt.plot(x, test_win_small_qrm_agent_1_moving_avg, color = 'black', label='Q-Learning Agent with RM (alpla=0.05) (gamma=0.99) (epsilon=0.7)')
    plt.plot(testing_episode_x, testing_win, color = 'black', label='Q-Learning with RM Agent')
    plt.legend(loc='lower left')
    plt.xlabel("Episode")
    plt.ylabel("Average Winrate")
    plt.title("Average winrate every 10k training episodes (small map)")
    plt.savefig("test-win-QRM-small-15", dpi=300, bbox_inches='tight')
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



    #save_q_table(env, "small-qrm.pkl")

    env.close()


def testing(number_of_testing_games, env, testing_scores, testing_steps_taken, testing_win):
        score = 0
        steps = 0
        wins = 0
        
        for i in range(number_of_testing_games):

            steps_taken = 0

            env.play = True
            while(env.play == True):
                env.step(use_greedy_strategy=True)

                steps_taken += 1

                #if(i % 1000 == 0):
                    #print(i)
            score += env.score
            steps += steps_taken
            wins += env.won

            env.reset()

        testing_scores.append(score/number_of_testing_games)
        testing_steps_taken.append(steps/number_of_testing_games)
        testing_win.append(wins/number_of_testing_games)


def save_q_table(self, filename="q_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(self.q_table, f)

def load_q_table(self, filename="q_table.pkl"):
    with open(filename, "rb") as f:
        self.q_table = pickle.load(f)

if __name__=="__main__":
    main()