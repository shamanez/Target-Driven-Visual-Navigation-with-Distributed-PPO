"""
A simple version of OpenAI's Proximal Policy Optimization (PPO). [https://arxiv.org/abs/1707.06347]

Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.

The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]

View more on my tutorial website: https://morvanzhou.github.io/tutorials

Dependencies:
tensorflow 1.8.0
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue

from scene_loader import THORDiscreteEnvironment as Environment

import pdb

EP_MAX = 1000
EP_LEN = 500
N_WORKER = 8                # parallel workers
GAMMA = 0.9                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0001               # learning rate for critic
MIN_BATCH_SIZE = 64         # minimum batch size for updating PPO
UPDATE_STEP = 15            # loop update operation n-steps
EPSILON = 0.2               # for clipping surrogate objective
GAME = 'CartPole-v0'

#env = gym.make(GAME)
#S_DIM = env.observation_space.shape[0]
#A_DIM = env.action_space.n

from constants import TASK_TYPE
from constants import TASK_LIST


class PPONet(object):
    def __init__(self):
        self.sess = tf.Session()
        #self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        self.tfs_S = tf.placeholder("float", [None, 2048,4], 'state_new')
        self.tfs_T = tf.placeholder("float", [None, 2048,4], 'target_new')

        self.tfs_S_N=tf.reshape(self.tfs_S, [-1, 8192])
        self.tfs_T_N=tf.reshape(self.tfs_T, [-1, 8192])



        # critic
        #w_init = tf.random_normal_initializer(0., .1)
        #lc = tf.layers.dense(self.tfs, 200, tf.nn.relu, kernel_initializer=w_init, name='lc')
        #self.v = tf.layers.dense(lc, 1)
        #self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        #self.advantage = self.tfdc_r - self.v
        ##self.closs = tf.reduce_mean(tf.square(self.advantage))
        #self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        #self.pi, pi_params,self.pi_new,self.v_new = self._build_anet('pi', trainable=True)
        pi_params,self.pi_new,self.v_new = self._build_anet('pi', trainable=True)

        #oldpi, oldpi_params,oldpi_new,oldv_new = self._build_anet('oldpi', trainable=False)
        oldpi_params,oldpi_new,oldv_new = self._build_anet('oldpi', trainable=False)
        
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]


        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v_new
        self.closs = tf.reduce_mean(tf.square(self.advantage))


        self.tfa = tf.placeholder(tf.int32, [None, ], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
        pi_prob = tf.gather_nd(params=self.pi_new, indices=a_indices)   # shape=(None, )
        oldpi_prob = tf.gather_nd(params=oldpi_new, indices=a_indices)  # shape=(None, )
        ratio = pi_prob/oldpi_prob
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.total_loss= self.aloss + 0.5*self.closs

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.total_loss)
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers

                data = np.vstack(data)

                #s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + 1].ravel(), data[:, -1:]

      

                s, t,a, r = data[:, :8192],data[:, 8192: 16384], data[:, 16384: 16384 + 1].ravel(), data[:, -1:]



                s=np.reshape(s,[s.shape[0],2048,4])
                t=np.reshape(t,[t.shape[0],2048,4])


                adv = self.sess.run(self.advantage, {self.tfs_S: s,self.tfs_T: t, self.tfdc_r: r})

                loss=self.sess.run(self.closs,{self.tfs_S: s,self.tfs_T: t, self.tfdc_r: r, self.tfa: a, self.tfadv: adv})

                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs_S: s,self.tfs_T: t, self.tfdc_r: r, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                #[self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]

                
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

    def _build_anet(self, name, trainable):

        with tf.variable_scope(name):
            with tf.variable_scope("Siamese", reuse=tf.AUTO_REUSE):
                self.siamese_s=self.construct_Siamese(self.tfs_S_N,trainable)
                self.siamese_t=self.construct_Siamese(self.tfs_T_N,trainable)
                self.concat=tf.concat(values=[self.siamese_s, self.siamese_t], axis=1)
                #self.obs=self.fusion_layer(self.concat,trainable)


            self.obs=self.fusion_layer(self.concat,trainable)


            #l_a = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            
            #a_prob = tf.layers.dense(l_a, A_DIM, tf.nn.softmax, trainable=trainable)


            #Newly added 
            l_a_new = tf.layers.dense(self.obs, 512, tf.nn.relu, trainable=trainable)
            a_prob_new = tf.layers.dense(l_a_new, 4, tf.nn.softmax, trainable=trainable)

            v_new = tf.layers.dense(l_a_new, 1,trainable=trainable)
            



        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        #return a_prob, params,a_prob_new,v_new
        return params,a_prob_new,v_new

    def construct_Siamese(self, input,trainable):
        layer_1 = tf.layers.dense(inputs=input, units=512, activation=tf.nn.leaky_relu, name='Siamese_layer_1',trainable=trainable)
        #layer_2=tf.layers.dropout(layer_1,rate=0.5,noise_shape=None,seed=None,training=True,name='Drop_out_1')
        #layer_3=tf.layers.dense(inputs=layer_2, units=128, activation=tf.nn.leaky_relu, name='Siamese_layer_2')
        return layer_1

    def fusion_layer(self, input,trainable): #This is also a shared fusion layer
        fuse_layer_1 = tf.layers.dense(inputs=input, units=512, activation=tf.nn.leaky_relu, name='Fuse_layer',trainable=trainable)
        #fuse_layer_2=tf.layers.dropout(fuse_layer_1,rate=0.5,noise_shape=None,seed=None,training=True,name='Drop_out_2')  #added dropout
        #fuse_layer_3 = tf.layers.dense(inputs=fuse_layer_1, units=128, activation=tf.nn.leaky_relu, name='Fuse_layer_3')
        return fuse_layer_1

    def choose_action(self,s_new,t_new):  # run by a local
        #prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: s[None, :]})


        prob_weights_new = self.sess.run(self.pi_new, feed_dict={self.tfs_S: [s_new],self.tfs_T:[t_new]})
        # action = np.random.choice(range(prob_weights.shape[1]),
        #                               p=prob_weights.ravel())  # select action w.r.t the actions prob


        action_new = np.random.choice(range(prob_weights_new.shape[1]),
                                      p=prob_weights_new.ravel())  # select action w.r.t the actions prob

 
        #return action,action_new
        return action_new


    
    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def get_v_new(self, S,T):
         #if s.ndim < 2: s = s[np.newaxis, :]
         return self.sess.run(self.v_new, {self.tfs_S: [S],self.tfs_T:[T]})[0, 0]


class Worker(object):
    def __init__(self, wid,target_id):
        self.wid = wid
        #self.env = gym.make(GAME).unwrapped
        self.env_new=Environment({'scene_name':'bathroom_02','terminal_state_id': int(target_id)})
        self.ppo = GLOBAL_PPO
        self.task_scope = target_id

     



    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        global GLOBAL_EP_new, GLOBAL_RUNNING_R_new, GLOBAL_UPDATE_COUNTER_new
        while not COORD.should_stop():
            #s = self.env.reset()
            self.env_new.reset()

            ep_r = 0
            ep_r_new = 0
            buffer_s, buffer_a, buffer_r = [], [], []

            buffer_s_new,buffer_t_new, buffer_a_new, buffer_r_new = [], [], [],[]


            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data
                    buffer_s_new, buffer_a_new, buffer_r_new = [], [], []
                a_new = self.ppo.choose_action(self.env_new.s_t, self.env_new.target) #This is suppose to take the optimal action

          
                # process game
                self.env_new.step(a_new)

                #s_, r, done, _ = self.env.step(a)


                r_new  = self.env_new.reward
                done_new = self.env_new.terminal
                r_new = 1 if done_new else -0.01

                

                # if done: r = -10
                # buffer_s.append(s)
                # buffer_a.append(a)
                # buffer_r.append(r-1)           # 0 for not down, -11 for down. Reward engineering

                buffer_s_new.append(self.env_new.s_t)
                buffer_t_new.append(self.env_new.target)
                buffer_a_new.append(a_new)
                buffer_r_new.append(r_new) 

                self.env_new.update()


                #s = s_
                s_new=self.env_new.s_t
                target=self.env_new.target
                #ep_r += r
                ep_r_new += r_new

                GLOBAL_UPDATE_COUNTER += 1         # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done_new:
                    if done_new:
                        #v_s_ = 0 
                        v_s_new=0                               # end of episode
                    else:
                        #v_s_ = self.ppo.get_v(s_)
                        v_s_new = self.ppo.get_v_new(s_new,target)
                    
                    # discounted_r = []                    # compute discounted reward
                    # for r in buffer_r[::-1]:
                    #     v_s_ = r + GAMMA * v_s_
                    #     discounted_r.append(v_s_)
                    # discounted_r.reverse()


                    discounted_r_new = []                    # compute discounted reward
                    for r in buffer_r_new[::-1]:
                        v_s_new = r_new + GAMMA * v_s_new
                        discounted_r_new.append(v_s_new)
                    discounted_r_new.reverse()

                    # bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, None]
  


                    # buffer_s, buffer_a, buffer_r = [], [], []
                    # QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue

    


                    bs_new,bt_new, ba_new, br_new = np.vstack([buffer_s_new]),np.vstack([buffer_t_new]), np.vstack(buffer_a_new), np.array(discounted_r_new)[:, None]
                    bs_new=np.reshape(bs_new,[bs_new.shape[0],-1])

                    bt_new=np.reshape(bt_new,[bs_new.shape[0],-1])


              

                    buffer_s_new,buffer_t_new, buffer_a_new, buffer_r_new = [], [], [],[]
                    QUEUE.put(np.hstack((bs_new, bt_new,ba_new, br_new)))          # put data in the queue



                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break
        
                    if done_new: break

            # record reward changes, plot later

         
            # if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_r)
            # else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+ep_r*0.1)
            # GLOBAL_EP += 1
            # print('{0:.1f}%'.format(GLOBAL_EP/EP_MAX*100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r,)

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R_new.append(ep_r_new)
            else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+ep_r_new*0.1)
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP/EP_MAX*100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r_new,)


        


if __name__ == '__main__':
    GLOBAL_PPO = PPONet()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out


    network_scope = TASK_TYPE
    list_of_tasks = TASK_LIST
    scene_scopes = list_of_tasks.keys()

    branches=[]

    for scene in scene_scopes:
        for task in list_of_tasks[scene]:
            branches.append((scene, task))

    NUM_TASKS = len(branches)

    workers = []

    for i in range(N_WORKER): #This is the parrele size

        scene, task = branches[i%NUM_TASKS]
        training_thread = Worker(wid=i,target_id=task)
        workers.append(training_thread)

    
    #workers = [Worker(wid=i,target_id=i) for i in range(N_WORKER)]
    

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    GLOBAL_RUNNING_R_new =[]
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()           # workers putting data in this queue
    threads = []
    for worker in workers:          # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()                   # training
        threads.append(t)




   
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update,))

  
    threads[-1].start()
    COORD.join(threads)

    # plot reward change and test
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode'); plt.ylabel('Moving reward'); plt.ion(); plt.show()
    #env = gym.make('CartPole-v0')

    self.env =Environment({'scene_name':'bathroom_02','terminal_state_id': int(26)})
    while True:
        self.env.reset()
        for t in range(1000):
            env.render()
            self.env.step(GLOBAL_PPO.choose_action(self.env.s_t, self.env.target))

            print(t)
            if done:
                break

