import sys
import gym
import torch
import pylab
import random
import csv
from collections import deque
from datetime import datetime
from copy import deepcopy
from skimage.transform import resize
from skimage.color import rgb2gray
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


def find_max_lifes(env):
    env.reset()
    _, _, _, info = env.step(0)
    return info['ale.lives']


def check_live(life, cur_life):
    if life > cur_life:
        return True
    else:
        return False


def pre_proc(X):
    x = np.uint8(resize(rgb2gray(X), (HEIGHT, WIDTH), mode='reflect') * 255)
    return x


def get_init_state(history, s):
    for i in range(HISTORY_SIZE):
        history[i, :, :] = pre_proc(s)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# approximate Q function using Neural Network
# state is input and Q Value of each action is output of network
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        return self.fc(x)


# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQNAgent():
    def __init__(self, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = True
        self.load_model = True

        # get size of action
        self.action_size = action_size
        self.MOMENTUM = 0.95  # Momentum used by RMSProp
        #self.MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.0001
        self.memory_size = 800000
        self.epsilon = 0.1
        self.epsilon_min = 0.02
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.batch_size = 32
        self.train_start = 10000
        self.update_target = 1000

        # create replay memory using deque
        self.memory = deque(maxlen=self.memory_size)

        # create main model and target model
        self.model = DQN(action_size)
        self.model.cuda()
        self.model.apply(self.weights_init)
        self.target_model = DQN(action_size)
        self.target_model.cuda()

        self.optimizer = optim.RMSprop(params=self.model.parameters(),lr=self.learning_rate)
        #self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model = torch.load('save_model/2019-04-12-22-34-20')

    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)
            print(m)
        elif classname.find('Conv') != -1:
            torch.nn.init.xavier_uniform(m.weight)
            print(m)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    #  get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state).unsqueeze(0)
            state = Variable(state).float().cuda()
            action = self.model(state).data.cpu().max(1)[1]
            return int(action)

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, history, action, reward, done):
        self.memory.append((history, action, reward, done))

    def get_sample(self, frame):
        mini_batch = []
        if frame >= self.memory_size:
            sample_range = self.memory_size
        else:
            sample_range = frame

        # history size
        sample_range -= (HISTORY_SIZE + 1)

        idx_sample = random.sample(range(sample_range), self.batch_size)
        for i in idx_sample:
            sample = []
            for j in range(HISTORY_SIZE + 1):
                sample.append(self.memory[i + j])

            sample = np.array(sample)
            mini_batch.append((np.stack(sample[:, 0], axis=0), sample[3, 1], sample[3, 2], sample[3, 3]))

        return mini_batch

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.get_sample(frame)
        mini_batch = np.array(mini_batch).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3]

        # bool to binary
        dones = dones.astype(int)

        # Q function of current state
        states = torch.Tensor(states)
        states = Variable(states).float().cuda()
        pred = self.model(states)

        # one-hot encoding
        a = torch.LongTensor(actions).view(-1, 1)

        one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_()
        one_hot_action.scatter_(1, a, 1)

        pred = torch.sum(pred.mul(Variable(one_hot_action).cuda()), dim=1)

        # Q function of next state
        next_states = torch.Tensor(next_states)
        next_states = Variable(next_states).float().cuda()
        next_pred = self.model(next_states).data.cpu()
        a_max_index=next_pred.argmax(1)
        next_pred_target=self.target_model(next_states).data.cpu()
        target=np.zeros(shape=self.batch_size)
        target=torch.from_numpy(target).float()
        for i in range(self.batch_size):
            target[i]=next_pred_target[i,a_max_index[i]]

        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Q Learning: get maximum Q value at s' from target model
        target = rewards + (1 - dones) * self.discount_factor * target
        target = Variable(target).cuda()

        self.optimizer.zero_grad()

        # MSE Loss function
        loss = F.smooth_l1_loss(pred, target)
        loss.backward()

        # and train
        self.optimizer.step()


if __name__ == "__main__":
    EPISODES = 500000
    HEIGHT = 84
    WIDTH = 84
    HISTORY_SIZE = 4

    env = gym.make('BreakoutDeterministic-v4')
    max_life = find_max_lifes(env)
    state_size = env.observation_space.shape
    actionspace=[0,2,3]
    action_size = 3
    scores, episodes = [], []
    agent = DQNAgent(action_size)
    recent_reward = deque(maxlen=100)
    frame = 0
    memory_size = 0
    for e in range(EPISODES):
        done = False
        score = 0

        history = np.zeros([5, 84, 84], dtype=np.uint8)
        step = 0
        d = False
        state = env.reset()
        life = max_life

        get_init_state(history, state)

        while not done:
            step += 1
            frame += 1
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(np.float32(history[:4, :, :]) / 255.)

            _,reward2,_,_=env.step(actionspace[action])
            next_state, reward, done, info =env.step(1)
            reward=reward+reward2
            pre_proc_next_state = pre_proc(next_state)
            history[4, :, :] = pre_proc_next_state
            ter = check_live(life, info['ale.lives'])

            life = info['ale.lives']
            r = np.clip(reward, -1, 1)

            # save the sample <s, a, r, s'> to the replay memory
            #agent.append_sample(deepcopy(pre_proc_next_state), action, r, ter)
            agent.append_sample(deepcopy(pre_proc_next_state), action, r, done)

            # every time step do the trainingF
            if frame >= agent.train_start:
                agent.train_model(frame)
                if frame % agent.update_target == 0:
                    agent.update_target_model()
            score += reward
            history[:4, :, :] = history[1:, :, :]

            # if frame % 50000 == 0:
            #     print('now time : ', datetime.now())
            #     scores.append(score)
            #     episodes.append(e)
            #     pylab.plot(episodes, scores, 'b')
            #     pylab.savefig("./save_graph/breakout_dqn.png")

            if done:
                recent_reward.append(score)
                # every episode, plot the play time
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon, "   steps:", step,
                      "    recent reward:", np.mean(recent_reward))
                if e % 20 == 0 :
                    scores.append(score)
                    episodes.append(e)
                if e % 500 == 0:
                    pylab.plot(episodes, scores, 'b')
                    pylab.savefig("./save_graph/breakout_dqn.png")
                    with open ("data.csv","w",newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        for i in scores:
                            writer.writerow([i])
                # if the mean of scores of last 10 episode is bigger than 400
                # stop training
                if e %1000==0:
                    torch.save(agent.model, "./save_model/"+time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
                if np.mean(recent_reward) > 50:
                    torch.save(agent.model, "./save_model/breakout_dqn")
                    #sys.exit()