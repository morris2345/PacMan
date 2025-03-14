import gymnasium as gym
from CustomPacMan import CustomPacManEnv
import time

def main():
    env = CustomPacManEnv(maze_size="normal")

    #print(env.grid)
    #while(env.play == True):
    for i in range(2500):
        env.step(use_greedy_strategy=False)
        #print(i)
        #print("--------------------------------------------------")
        #print(env.grid)
        #print(env.pill_active)
        #env.render()
        #time.sleep(0.25)

    for i in range(2500):
            env.step(use_greedy_strategy=True)
            #print("--------------------------------------------------")
            #print(env.grid)
            #print(env.pill_active)
            env.render()
            time.sleep(0.25)
    #print(env.grid)


    env.close

if __name__=="__main__":
    main()