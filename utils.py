import math
import matplotlib.pyplot as plt
import torch
from parameters import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
plt.style.use('ggplot')


def preprocess_frame(frame):
    # frame = frame.copy()
    # frame = frame.transpose((2, 0, 1))  # Torch image format
    frame = torch.from_numpy(frame)
    frame = frame.to(device, dtype=torch.float32)
    # frame = frame.unsqueeze(1)
    frame = frame.unsqueeze(0)

    return frame


def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay)
    return epsilon


def plot_results(rewards_total):
    plt.figure(figsize=(12, 5))
    plt.title("Rewards")
    plt.xlabel("# Epsiodes")
    plt.plot(rewards_total, alpha=0.6, color='red')
    plt.savefig("Pong-results.png")
    plt.close()


def print_report(episode, report_interval, rewards_total, mean_reward_100, epsilon, frames_total):
    print("\n*** Episode %i *** "
          "\nAv.reward: [last %i]: %.2f, [last 100]: %.2f, [all]: %.2f "
          "\nepsilon: %.2f, frames_total: %i"
          % (episode, report_interval,
             sum(rewards_total[-report_interval:]) / report_interval,
             mean_reward_100,
             sum(rewards_total) / len(rewards_total),
             epsilon, frames_total)
          )


def load_model():
    return torch.load(file2save)


def save_model(model):
    torch.save(model.state_dict(), file2save)