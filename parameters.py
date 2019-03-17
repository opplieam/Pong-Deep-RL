num_episodes = 600
report_interval = 20

replay_mem_size = 100000
batch_size = 32

update_target_frequency = 10000

egreedy = 0.9
egreedy_final = 0.02
egreedy_decay = 10000

learning_rate = 0.0001
gamma = 0.99

file2save = 'pong_save.pth'
video_directory = './PongVideos/'