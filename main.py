import gym
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import *
from keras.applications import VGG16


from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

env = gym.make('MsPacman-v0')
def collect_state(env,num_samples = 100000):
    env.reset()
    try:
        for _ in range(num_samples):
            env.render()
            a = env.action_space.sample()
            obs,r,d,i = env.step(a)
            np.save('frame-%d.npy' % _,obs )
            if d: env.reset()
    except KeyboardInterrupt:
        env.reset()
        return

def get_encoder(scale = .5,encode_dim = 120):
    h = int(210 * scale)
    w = int(160 * scale)
    inp = Input(shape = (h,w,3))
    vgg = VGG16(input_shape = (h,w,3),classes = encode_dim,include_top = False)(inp)
    vgg.trainable = False
    enc = Flatten()(vgg)
    model = Model(inp,enc)
    model.trainable_weights = []
    model.compile(loss = 'mse',optimizer = 'rmsprop')
    model.summary()
    return inp,enc,model

def get_actor():
    img_inp,encoder_out,encoder_model = get_encoder(encode_dim = 300)
    act = Dense(750,activation = 'relu')(encoder_out)
    act = Dense(350,activation = 'relu')(act)
    act = Dense(9,activation = 'softmax')(act)

    actor = Model(img_inp,act)
    actor.compile(loss = 'mse',optimizer = 'rmsprop')

    print actor.summary()
    return actor

def train_pacman(lr = 0.0001,epoch = 50000):
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)

_,__,actor = get_encoder()
X = np.stack([np.zeros((105,80,3)) for _ in range(1000)])
y = np.stack([np.zeros((3072,)) for _ in range(1000)])
actor.fit(X,y)
