import gym

from atari_wrappers import wrap_deepmind, make_atari
from qnet_agent import QNet_Agent
from exp_replay import ExperienceReplay
from utils import plot_results, calculate_epsilon, print_report
from parameters import *


env = make_atari('PongNoFrameskip-v4')
env = wrap_deepmind(env=env, frame_stack=True, pytorch_img=True)
env = gym.wrappers.Monitor(
    env, video_directory, video_callable=lambda episode_id: episode_id % 50 == 0,
    force=True
)

# seed_value = 1
# env.seed(seed_value)
number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.n


memory = ExperienceReplay(capacity=replay_mem_size)
qnet_agent = QNet_Agent(
    number_of_outputs, learning_rate, env, memory, batch_size, gamma,
    update_target_frequency
)

rewards_total = []
solved_after = 0
frames_total = 0
for episode in range(num_episodes):
    state = env.reset()
    score = 0
    while True:
        frames_total += 1
        epsilon = calculate_epsilon(frames_total)
        action = qnet_agent.select_action(state, epsilon)
        new_state, reward, done, info = env.step(action)
        score += reward

        memory.push(state, action, new_state, reward, done)
        qnet_agent.optimize()
        # env.render()
        state = new_state
        if done:
            solved_after = episode
            rewards_total.append(score)
            plot_results(rewards_total)
            mean_reward_100 = sum(rewards_total[-100:]) / 100
            if episode % report_interval == 0 and episode > 0:
                print_report(episode, report_interval, rewards_total,
                             mean_reward_100, epsilon, frames_total)
            break

print("Average reward: %.2f" % (sum(rewards_total)/num_episodes))
print("Average reward (last 100 episodes): ", (sum(rewards_total[-100:])/100))
print("Solved after %i episodes" % solved_after)
env.close()
