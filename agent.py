import flappy_bird_gymnasium
import gymnasium as gym
import torch
env = gym.make("FlappyBird-v0", render_mode="human")

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

state, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Processing:
    next_state, reward, terminated, _, info = env.step(action)
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()