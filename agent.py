import flappy_bird_gymnasium
import gymnasium as gym
import torch
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch.nn as nn
import torch.optimizer as optim

if torch.cuda.is_available():
    device="cuda"
    print("yes")
else:
    device="cpu"

class Agent():

    def __init__(self,param_set):
        self.param_set = param_set

        with open("paramters.yaml","r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha              = params["alpha"]
        self.gamma              = params["gamma"]
        self.epsilon_init       = params["epsilon_init"]
        self.epsilon_min        = params["epsilon_min"]
        self.epsilon_decay      = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size    = params["mini_batch_size"]

        self.network_sync_rate  = params["network_sync_rate"]
        self.reward_threshold   = params["reward_threshold"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None # Might use adam


    def run(self,is_training=True,render=False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states,num_actions).to(device)

        

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states,num_actions).to(device)
            #copy the weight and bias value from policy to target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps =0
            self.optimizer = optim.Adam(policy_dqn.parameters(),lr = self.alpha)

        for episode itertools.count():
            
            state, _ = env.reset()
            state = torch.tensor(state,dtype = torch.float,device=device)
            episode_rewards = 0
            terminated = False

            while not terminated:
                if is_training and random.random()<epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action,dtype = torch.long,device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                next_state, reward, terminated, _, info = env.step(action.item())


                reward = torch.tensor(reward,dtype = torch.float,device=device)
                next_state = torch.tensor(next_state,dtype = torch.float,device=device)
                
                if is_training:
                    memory.append(state,action,next_state,reward,terminated)
                    steps+=1

                state = next_state
                episode_rewards+=reward

            print(f"for episode = {episode+1} having total reward = {episode_rewards} and epsilon = {epsilon}")
            if is_training:
                #epsilon decay
                epsilon = max(epsilon *self.epsilon_decay,self.epsilon_min)

            if is_training and len(memory) >self.mini_batch_size:
                #sample 
                mini_batch = memory.sample(self.mini_batch_size)

                optimize(mini_batch,policy_dqn,target_dqn)

                #Sync the network
                if steps >self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0
            # env.close()
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # get batch of experiences
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        # calculate target Q-values
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        # calculate current Q-values
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # compute loss
        loss = self.loss_fn(current_q, target_q)

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()