import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
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
    
    #TODO
    # add a trainer

    class Trainer:
        def __init__(self, model, lr):
            self.lr = lr
            self.model = model
            self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
            self.criterion = nn.MSELoss()
        
        def train(self, observation, search_policies, returns):
            observation = torch.from_numpy(observation)
            search_policies = torch.from_numpy(search_policies)
            returns = torch.from_numpy(returns)

            self.optimizer.zero_grad()
            logits, policy, value = self.step_model(observation)
            

            # policy_loss = ... research what this loss is and why it's like this
            ## policy_loss = 5*torch.mean(torch.sum(-search_pis * logsoftmax(logits), dim=1))

            value_loss = self.value_criterion(value, returns)
            loss = policy_loss + value_loss
            loss.backward()
            self.optimizer.step()  

    # add a logger function