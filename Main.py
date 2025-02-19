import gymnasium as gym
from CustomPacMan import CustomPacManEnv

def main():
    print("hey there")
    env = CustomPacManEnv(maze_size="normal")

    print(env.grid)

    env.close

if __name__=="__main__":
    main()