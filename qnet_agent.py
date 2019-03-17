import torch

from deep_network import NeuralNetwork
from utils import preprocess_frame, save_model

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.manual_seed(1)


class QNet_Agent(object):
    save_model_frequency = 10000
    number_of_frames = 0

    def __init__(self, number_of_outputs, learning_rate, env, memory, batch_size, gamma, update_target_frequency):
        self.env = env
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_target_frequency = update_target_frequency
        self.nn = NeuralNetwork(number_of_outputs).to(device)
        self.target_nn = NeuralNetwork(number_of_outputs).to(device)
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.nn.parameters(),
                                          lr=learning_rate)

    def select_action(self, state, epsilon):
        random_for_egreedy = torch.rand(1).item()

        if random_for_egreedy > epsilon:
            with torch.no_grad():
                state = preprocess_frame(state)
                action_from_nn = self.nn(state)
                action = torch.max(action_from_nn, 1)[1].item()
        else:
            action = self.env.action_space.sample()

        return action

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, new_state, reward, done = self.memory.sample(self.batch_size)
        state = [preprocess_frame(frame) for frame in state]
        state = torch.cat(state)  # stack tensor

        new_state = [preprocess_frame(frame) for frame in new_state]
        new_state = torch.cat(new_state)

        reward = torch.Tensor(reward).to(device)
        action = torch.LongTensor(action).to(device)
        done = torch.Tensor(done).to(device)

        # Double DQN
        max_new_state_indexes = torch.argmax(self.nn(new_state).detach(), 1)

        new_state_values = self.target_nn(new_state).detach()
        max_new_state_values = new_state_values.gather(
            1, max_new_state_indexes.unsqueeze(1)
        ).squeeze(1)

        target_value = reward + (1 - done) * self.gamma * max_new_state_values
        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(
            1)

        loss = self.loss_func(predicted_value, target_value)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.nn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.number_of_frames % self.save_model_frequency == 0:
            save_model(self.nn)

        if self.number_of_frames % self.update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())
        self.number_of_frames += 1
