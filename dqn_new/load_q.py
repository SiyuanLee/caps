import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

learning_starts = int(1e5)
interval = int(1e5)
num_timesteps = int(104e5)
aver_q = []

for t in range(learning_starts, num_timesteps, interval):
    pkl_file = open('/home/lsy/PycharmProjects/ple-monstrerkong/examples/dqn_new/logs/change_learning_rate_12_04_17_21:45:05/'+str(t)+'_q.pkl', 'rb')
    q = pickle.load(pkl_file)
    pkl_file.close()
    Q_t = []
    for room in range (5):
        Q_t.append(np.mean(np.array(q[room]).max(axis=1)))
    aver_q.append(Q_t)
    # pkl_file = open('logs/change_learning_rate_12_04_17_21:45:05/' + str(t) + '_q.pkl', 'rb')

aver_q = np.array(aver_q)

[T, Room] =np.shape(aver_q)
for room in range(Room):
    plt.plot(aver_q[:, room], color=np.random.rand(3), label=str(room))
    plt.grid()
    plt.legend(loc='best')
    plt.savefig('change_learning_rate.png')
