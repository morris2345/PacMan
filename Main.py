import gymnasium as gym
from CustomPacMan import CustomPacManEnv
import time
import pickle

def main():
    env = CustomPacManEnv(maze_size="small", algo="QL")

    load_q_table(env, "small-ql-newVersion.pkl")
    print(len(env.q_table))
    #print(env.grid)
    for i in range(500000):
        env.play = True
        while(env.play == True):
            env.step(use_greedy_strategy=False)
            if(i % 10000 == 0):
                print(i)
                #env.render()
                #time.sleep(0.25)
        #print(i)
        #print("--------------------------------------------------")
        #print(env.grid)
        #print(env.pill_active)
        #env.render()
        #time.sleep(0.25)

    for i in range(3):
        env.play = True
        while(env.play == True):
            env.step(use_greedy_strategy=True)
        #print("--------------------------------------------------")
        #print(env.grid)
        #print(env.pill_active)
            env.render()
            time.sleep(0.25)
    #print(env.grid)
    print(len(env.q_table))
    save_q_table(env, "small-ql-newVersion.pkl")

    env.close

def save_q_table(self, filename="q_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(self.q_table, f)

def load_q_table(self, filename="q_table.pkl"):
    with open(filename, "rb") as f:
        self.q_table = pickle.load(f)

if __name__=="__main__":
    main()