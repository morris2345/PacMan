import gymnasium as gym
from CustomPacMan import CustomPacManEnv

def main():
    env = CustomPacManEnv(maze_size="normal")

    print(env.grid)

    env.step()
    print("--------------------------------------------------")
    print(env.grid)

    env.step()
    print("--------------------------------------------------")
    print(env.grid)


    env.close

if __name__=="__main__":
    main()