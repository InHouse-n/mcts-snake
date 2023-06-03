import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from IPython import display

class DeepNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.dense_policy = nn.Linear(hidden_size, output_size)
        self.dense_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.dense(x))
        logits = self.dense_policy(x)
        policy = F.softmax(logits, dim=1)

        value = self.dense_value(x)
        return logits, policy, value
    
    def predict(self, observation):
        
        x = torch.from_numpy(observation)
        _, policy, value = self.forward(x)

        return policy.detach().numpy(), value.detach().numpy()
    

    class Trainer:
        def __init__(self, model, lr):
            self.lr = lr
            self.model = model
            self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
            self.criterion = nn.MSELoss()
            self.entropy_criterion = nn.CrossEntropyLoss()
        
            def train(self, observation, search_policies, rewards):
                observation = torch.from_numpy(observation)
                search_policies = torch.from_numpy(search_policies)
                rewards = torch.from_numpy(rewards)

                self.optimizer.zero_grad()
                logits, _, value = self.model(observation)
                
                
                logsoftmax = nn.LogSoftmax(dim=1)
                policy_loss = 5 * torch.mean(torch.sum(-search_policies
                                                * logsoftmax(logits), dim=1))

                value_loss = self.criterion(value, rewards)
                loss = policy_loss + value_loss
                loss.backward()
                self.optimizer.step()

                return value_loss.data.numpy(), policy_loss.data.numpy()

            self.train = train
    
    class TrainingMemory:
        def __init__(self, batch_size=200):
            self.batch_size = batch_size
            self.memory = []
            self.training_sample = []
        def remember(self, observation, search_policy, reward):
            self.memory.append((observation, search_policy, reward))
        
        def get_training_sample(self):
            if len(self.memory) > self.batch_size:
                self.training_sample = random.sample(self.memory, self.batch_size)
            else:
                self.training_sample = self.memory
            
            observations, search_policies, rewards = zip(*self.training_sample)

            return observations, search_policies, rewards
         


    class TrainingLogger:
        def __init__(self):
            plt.ion()
            self.iterations = []
            self.value_losses = []
            self.policy_losses = []
            self.rewards = []
            
        def add_info(self, iteration, value_loss, policy_loss, reward):
            self.iterations.append(iteration)
            self.value_losses.append(value_loss)
            self.policy_losses.append(policy_loss)
            self.rewards.append(reward)
        
        def plot(self, save_to=False):
            display.clear_output(wait=True)
            display.display(plt.gcf())
            plt.clf()
            plt.title("Training...")
            plt.xlabel("Number of Games")
            plt.ylabel("losses")
            plt.plot(self.value_losses)
            plt.plot(self.policy_losses)
            plt.ylim(ymin=0)
            plt.text(len(self.value_losses) - 1, self.value_losses[-1], str(self.value_losses[-1]))
            plt.text(len(self.policy_losses) - 1, self.policy_losses[-1], str(self.policy_losses[-1]))
            if save_to:
                plt.savefig(save_to, bbox_inches='tight')
            plt.show(block=False)
            plt.pause(0.1)