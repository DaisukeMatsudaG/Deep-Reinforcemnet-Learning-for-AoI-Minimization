!nvidia-smi
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
%cd /content
!mkdir drive
%cd drive
!mkdir MyDrive
%cd ..
%cd ..
!google-drive-ocamlfuse /content/drive/MyDrive
# -*- coding: utf-8 -*-
"""again_3times_same_start_Nposi_K_90_5_1_0e_5_SN9_Multi.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1zplywjsU7j7_3mMa1Py8-iweq83ulHpd
"""
import os
#from google.colab import files
#from google.colab import drive
#drive.mount('/content/drive')
#import wandb
!pip install tensorflow==2.5.0
import tensorflow as tf
print('TensorFlow', tf.__version__)
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam
import gym
import os
import argparse
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float32')
#wandb.init(name='DQN', project="deep-rl-tf2")

parser = argparse.ArgumentParser()
parser.add_argument('--amount_SN', type=int, default=8)
parser.add_argument('--gamma', type=float, default=0.90)
parser.add_argument('--train_ep', type=int, default=9000)
#parser.add_argument('--lr', type=float, default=0.00025)
parser.add_argument('--lr', type=float, default=1.0e-4)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.9999)
parser.add_argument('--eps_min', type=float, default=0.10)
#parser.add_argument('--number_SN', type=int, default=amount_SN)

args = parser.parse_args(args=[])

amount_SN = args.amount_SN
#file_name = 'sensing_per_5_0_00025_New_second_positions_SN9'
file_name = '1_0_e_4'
#file_name = 'check_test'
result_dir = '/content/drive/MyDrive/modi_env/Single_DQN/SN'+str(args.amount_SN)+'/'+str(file_name)


class AoI(gym.Env): # (1)
    # ???????????????
    amount_SN = amount_SN#???????????????
    max_AoI = 100
    x_p = 0
    x_m = 1
    y_p = 2
    y_m = 3

        #?????????
    def __init__(self):
        super(AoI, self).__init__()
        # ?????????????????????
        self.agent_pos = [200, 200]
        self.max_AoI = self.max_AoI
        self.amount_SN = self.amount_SN
        #self.SNposition_list_base=[[5,5],[-2+3,-2+3],[2+3,-2+3],[-2+3,2+3],[6+3,-2+3],[6+3,2+3],[2+3,6+3],[6+3,6+3],[-2+3,6+3]]
        #self.SNposition_list_base=[[5-1,5-1],[1-1,1-1],[5-1,1-1],[1-1,5-1],[9-1,1-1],[9-1,5-1],[5-1,9-1],[9-1,9-1],[1-1,9-1]]
        self.SNposition_list_base=[[8,4],[4,0],[0,0],[8,0],[0,4],[4,8],[8,8],[0,8],[2,6],[6,2],[6,6],[2,2]]
        self.SNposition_list=[[50*self.SNposition_list_base[i][0],50*self.SNposition_list_base[i][1]] for i in range(self.amount_SN)]
        #self.SN_distribution_list=[1,2,3,4,1,2,3,4,1]
        self.SN_distribution_list=[1,2,3,4,1,2,3,4,1,2,3,4]
        self.time_slot=0
        self.done=False
        #self.seconds_count=0
        ####?????????????????????????????????????????????
        x =  np.arange(0, 100, 1)
        y1= [exp_dist(2/1,i) for i in x]
        y2= [exp_dist(2/10,i) for i in x]
        y3= [exp_dist(2/20,i) for i in x]
        y4= [exp_dist(2/40,i) for i in x]
        self.exponential_distribution_list=[y1,y2,y3,y4]
        ####?????????????????????????????????????????????

        # ?????????????????????????????????????????? (2)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=321, shape=(1+2+self.amount_SN,))

    # ????????????????????? (3)
    def reset(self):
        self.time_slot=0
        self.done=False

        self.agent_pos_1 = [200, 200]#q_1(-1+1)

        self.S_list=[0 for iS in range(self.amount_SN)]#S_1(-1+1)

        self.AoI_list=[1 for iA in range(self.amount_SN)]#A(-1+1)

        self.big_o_list=[0 for iO in range(self.amount_SN)]#O(-1+1)

        self.o_list=[1 for io in range(self.amount_SN)]#o(-1+1)
        self.previous_time_slot_list=[0 for io in range(self.amount_SN)]
        
        self.U_list=[1 for iU in range(self.amount_SN)]#U(-1+1)

        self.state_list = [self.time_slot,self.agent_pos[0], self.agent_pos[1]]
        for i in range(self.amount_SN):
          self.state_list.append(self.AoI_list[i])

        self.state = np.array(self.state_list)

        return self.state

      # ?????????1?????????????????? (3)
    def step(self, action):
        #agent1??????????????????????????????S(n)
        for iS in range(self.amount_SN):
          if (self.agent_pos[0]==self.SNposition_list[iS][0] and self.agent_pos[1]==self.SNposition_list[iS][1] and self.U_list[iS] >0):
            self.S_list[iS] = 1
          else:
            self.S_list[iS] = 0

        for iO in range(self.amount_SN):
          self.big_o_list[iO] = min((self.big_o_list[iO]+self.o_list[iO]),1)-self.S_list[iO]#?????????S(n)?????????not S(n+1)
        
        self.amount_AoI = sum(self.AoI_list)
        self.reward = self.amount_AoI / self.amount_SN


        self.time_slot += 1##?????????????????????????????????????????????????????????????????????
        #??????????????????n+1?????????????????????????????????a(n)???agent(n)?????????
        

        if not self.done:
            #??????????????????n+1?????????????????????????????????a(n)???agent(n)?????????
            if action == self.x_p:
                self.agent_pos[0] = self.agent_pos[0] + 50
            elif action == self.x_m:
                self.agent_pos[0] = self.agent_pos[0] - 50
            elif action == self.y_p:
                self.agent_pos[1] = self.agent_pos[1] + 50
            elif action == self.y_m:
                self.agent_pos[1] = self.agent_pos[1] - 50
            else:
                raise ValueError("Received invalid action={}".format(action))

                #?????????next_step???AoI????????????????????????
        for iA in range(self.amount_SN):
          if self.S_list[iA]==1:
            self.AoI_list[iA]=self.U_list[iA]#?????????A(n+1)=U(n)
          else:
            self.AoI_list[iA]=self.AoI_list[iA]+1##?????????A(n+1)=A(n)+1
        
        ##??????????????????????????????????????????????????????????????????????????????
        #IoT?????????????????????????????????
          #?????????????????????
          #?????????????????????????????????????????????
          #if??????????????????????????????????????????????????????????????????????????????
        for io in range(self.amount_SN):
          #?????????
          self.probability_for_exponential_distribution=np.random.random()
          #????????????????????????[???????????????y1~y4][???????????????????????????]
          #[???????????????????????????]=[??????????????????????????? ??? previous????????????????????????]
          #????????????????????????[???????????????y1~y4][??????????????????????????? ??? previous????????????????????????]
          #??????????????????????????????if?????????o_list?????????
          self.threshold_probability = self.exponential_distribution_list[self.SN_distribution_list[io]-1][self.time_slot-self.previous_time_slot_list[io]]
          if self.probability_for_exponential_distribution < self.threshold_probability:
            self.o_list[io]=1
            self.previous_time_slot_list[io]=self.time_slot
            #timeslot????????????????????????????????????
          else:
            self.o_list[io]=0

        #for io in range(self.amount_SN):
        #  if self.action_step % 5==0:# or self.S_list[io]==1:#10????????????????????????????????????????????????????????????UAV????????????????????????
        #    self.o_list[io]=1
        #  else:
        #    self.o_list[io]=0

        for iU in range(self.amount_SN):
          if (self.big_o_list[iU]==0 and self.o_list[iU]==0):#O(n)=0??????o(n+1)=0
            self.U_list[iU]=0
          elif self.o_list[iU]==1:
            self.U_list[iU]=1
          else:
            self.U_list[iU]= self.U_list[iU] + 1

        for iA in range(self.amount_SN):
          self.AoI_list[iA] = np.clip(self.AoI_list[iA], 0, self.max_AoI )

        self.done = self.time_slot == 100##100????????????????????????????????????????????????

        self.state_list=[self.time_slot, self.agent_pos[0],self.agent_pos[1]]

        for i in range(self.amount_SN):
          self.state_list.append(self.AoI_list[i])

        self.next_state = np.array(self.state_list)

        return self.next_state, self.reward, self.done,self.U_list, {}

def exp_dist(lambda_, x):

    return 1-np.exp(- lambda_*x)
        #power consumption


def res_actions(x,y):
  res_actions_list = [0,1,2,3]
  if x == 400:
    res_actions_list.remove(0)
  elif x == 0:
    res_actions_list.remove(1)
  else:
    pass
  if y == 400:
    res_actions_list.remove(2)
  elif y == 0:
    res_actions_list.remove(3)
  else:
    pass
  res_actions_list = np.array(res_actions_list)
  #print(res_actions_list)
  return res_actions_list
#targets[range(args.batch_size), actions]
def res_output_layers(outputs,res_actions_list):
  res_outputs = np.empty((2,len(res_actions_list)))
  #print('?????????',res_outputs)
  for ri in range(len(res_actions_list)):
    res_outputs[0][ri] = outputs[res_actions_list[ri]]
    res_outputs[1][ri] = res_actions_list[ri]
  #print('?????????',res_outputs)  
  return res_outputs

def res_next_output_layers(next_outputs,res_next_actions_lists):
  min_output_batch = []
  for ni in range(len(next_outputs)):
    res_next_outputs = []
    for nj in range(len(res_next_actions_lists[ni])):
      res_next_outputs.append(next_outputs[ni][res_next_actions_lists[ni][nj]])
    min_output_batch.append(np.min(res_next_outputs))
  #print(min_output_batch)
  return np.array(min_output_batch)
    #res_next_outputs_batch.append(res_next_outputs)
  #print(len(res_next_outputs_batch))
  #min_output_list = []
  #min_output_list
    
def choose_res_action(res_outputs):
  argmin_number = np.argmin(res_outputs[0])
  min_action = int(res_outputs[1][argmin_number])
  return min_action

class ReplayBuffer:
    result_dir = result_dir
    def __init__(self, capacity=40000):
        self.result_dir = result_dir
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state, action, reward, next_state, done,res_actions_list):
        self.buffer.append([state, action, reward, next_state, done,res_actions_list])
    
    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done,res_actions_lists = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        res_actions_lists = np.array(res_actions_lists).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done, res_actions_lists
    
    def max_sample(self):
        max_sample = self.buffer
        states, actions, rewards, next_states, done,res_actions_lists = map(np.asarray, zip(*max_sample))
        states = np.array(states).reshape(self.size(), -1)
        next_states = np.array(next_states).reshape(self.size(), -1)
        res_actions_lists = np.array(res_actions_lists).reshape(self.size(), -1)
        return states, actions, rewards, next_states, done, res_actions_lists

    def size(self):
        return len(self.buffer)


    def save_buffer(self,episode_number):
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        f = open(os.path.join(self.result_dir,'replay_buffer_check_point_'+str(episode_number)+'_data_user.txt'),'a')
        f.write(str(self.buffer))
        f.close()    

class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim  = state_dim
        self.action_dim = aciton_dim
        self.epsilon = args.eps
        
        self.model = self.create_model()
        self.buffer = ReplayBuffer()

    def create_model(self):
        model = tf.keras.Sequential([
            InputLayer((self.state_dim,)),
            Dense(200, activation='relu'),
            Dense(200, activation='relu'),
            Dense(200, activation='relu'),
            Dense(self.action_dim)
        ])
        model.compile(loss='mse',metrics='mae', optimizer=Adam(args.lr))
        return model
    def predict_on_batch(self,state):##?????????predict??????????????????????????????????????????????????????????????????
        return self.model.predict_on_batch(state)#predict_on_batch???????????????????????????

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        res_actions_list = res_actions(state[0][1],state[0][2])
        q_value = self.predict_on_batch(state)[0]#???????????????predict????????????????????????????????????????????????
        if np.random.random() < self.epsilon:
            return random.choice(res_actions_list)##?????????restrict???????????????????????????????????????????????????????????????????????????
        else:
            res_outputs = res_output_layers(q_value,res_actions_list)

            action = choose_res_action(res_outputs)
            return action#argmin?????????????????????????????????]

    def test_get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        res_actions_list = res_actions(state[0][1],state[0][2])
        q_value = self.predict_on_batch(state)[0]#???????????????predict????????????????????????????????????????????????        
        res_outputs = res_output_layers(q_value,res_actions_list)

        action = choose_res_action(res_outputs)
        return action#argmin?????????????????????????????????

    def train(self, states, targets):#?????????fit????????????????????????????????????????????????
        #self.model.fit(states, targets, batch_size=args.batch_size, epochs=1, verbose=1)
        loss, td_error = self.model.train_on_batch(states, targets)

        return loss, td_error

    def evaluate_model(self,states, targets):
        results = self.model.evaluate(states, targets,batch_size=len(states))

        return results

class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim  = state_dim
        self.action_dim = aciton_dim
        self.epsilon = args.eps
        
        self.model = self.create_model()
        self.buffer = ReplayBuffer()

    def create_model(self):
        model = tf.keras.Sequential([
            InputLayer((self.state_dim,)),
            Dense(200, activation='relu'),
            Dense(200, activation='relu'),
            Dense(200, activation='relu'),
            Dense(self.action_dim)
        ])
        model.compile(loss='mse',metrics='mae', optimizer=Adam(args.lr))
        return model
    def predict_on_batch(self,state):##?????????predict??????????????????????????????????????????????????????????????????
        return self.model.predict_on_batch(state)#predict_on_batch???????????????????????????

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        res_actions_list = res_actions(state[0][1],state[0][2])
        q_value = self.predict_on_batch(state)[0]#???????????????predict????????????????????????????????????????????????
        if np.random.random() < self.epsilon:
            return random.choice(res_actions_list)##?????????restrict???????????????????????????????????????????????????????????????????????????
        else:
            res_outputs = res_output_layers(q_value,res_actions_list)

            action = choose_res_action(res_outputs)
            return action#argmin?????????????????????????????????]

    def test_get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        res_actions_list = res_actions(state[0][1],state[0][2])
        q_value = self.predict_on_batch(state)[0]#???????????????predict????????????????????????????????????????????????        
        res_outputs = res_output_layers(q_value,res_actions_list)

        action = choose_res_action(res_outputs)
        return action#argmin?????????????????????????????????

    def train(self, states, targets):#?????????fit????????????????????????????????????????????????
        #self.model.fit(states, targets, batch_size=args.batch_size, epochs=1, verbose=1)
        loss, td_error = self.model.train_on_batch(states, targets)

        return loss, td_error

    def evaluate_model(self,states, targets):
        results = self.model.evaluate(states, targets,batch_size=len(states))

        return results

class Agent:
    result_dir = result_dir
    def __init__(self, env):
        self.result_dir = result_dir
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()

        self.file_name = file_name

        self.buffer = ReplayBuffer()
    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)
    
    def replay(self):
        for _ in range(1):
            states, actions, rewards, next_states, done,res_next_actions_lists = self.buffer.sample()
            targets = self.target_model.predict_on_batch(states)
            next_q_values = self.target_model.predict_on_batch(next_states)
            future_return = res_next_output_layers(next_q_values,res_next_actions_lists)

            targets[range(args.batch_size), actions] = rewards + (1-done) * future_return * args.gamma
            loss, td_error = self.model.train(states, targets)
        return loss, td_error

    def evaluate_by_buffer(self):
        states, actions, rewards, next_states, done,res_next_actions_lists = self.buffer.max_sample()
        targets = self.target_model.predict_on_batch(states)
        #q_preds = targets#???????????????????????????????????????????????????????????????????????????
        next_q_values = self.target_model.predict_on_batch(next_states)
        future_return = res_next_output_layers(next_q_values,res_next_actions_lists)
        targets[range(len(future_return)), actions] = rewards + (1-done) * future_return * args.gamma
        results = self.model.evaluate_model(states, targets)

        return results    

    def train(self, max_episodes=10000):
        ep_list = []
        ep_loss_list = []
        loss_list = []
        mae_list = []
        reward_list = []
        total_reward_list = []
        average_AoI_list = []
        total_average_AoI_list = []
        for ep in range(max_episodes):
            total_reward = 0
            total_average_AoI = 0
            done = False
            state = self.env.reset()
            step_count=0
            while not done:
                AoI_list=[int(state[3+i]) for i in range(amount_SN)]
                average_AoI = sum(AoI_list) / amount_SN
                action = self.model.get_action(state)
                next_state, reward, done, _,U_list = self.env.step(action)
                res_next_actions_list = res_actions(next_state[1],next_state[2])
                self.buffer.put(state, action, reward, next_state, done,res_next_actions_list)#reward??0.01???????????????????????????
                state = next_state
                total_reward += reward
                total_average_AoI += average_AoI
                step_count+=1
                if self.buffer.size() >= args.batch_size:
                    loss, td_error = self.replay()
            ep_list.append(ep)    
            reward_list.append(reward)
            total_reward_list.append(total_reward/step_count)
            average_AoI_list.append(average_AoI)
            total_average_AoI_list.append(total_average_AoI/step_count)
            self.target_update()
            if ep > 0:
                AoI_list=[int(state[3+i]) for i in range(amount_SN)]
                print('ep{} loss={} mae={} F_AoI/U={}/{} position={} taA={} TR={}'\
                      .format(ep, loss, td_error,\
                              AoI_list, U_list, [state[1],state[2]],\
                              total_average_AoI/step_count ,total_reward/step_count))
                if (ep+1) % 100 == 0:
                  results = self.evaluate_by_buffer()
                  ep_loss_list.append(ep)
                  loss_list.append(results[0])
                  mae_list.append(results[1])
                  print('loss',results[0],'mae',results[1])
                
        ##???????????????????????????
        #results???????????????????????????
        #result_dir = 'results'
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        fig1 = plt.figure()
        plt.plot(ep_list, total_reward_list);
        plt.show()
        fig1.savefig(os.path.join(self.result_dir,str(self.file_name)+'_reward.png'))
        fig2 = plt.figure()
        plt.plot(ep_loss_list,loss_list);
        plt.show()
        fig2.savefig(os.path.join(self.result_dir, str(self.file_name)+'_loss.png'))
        self.target_model.model.save_weights(os.path.join(self.result_dir, str(self.file_name)+'_model.h5'))
        f = open(os.path.join(self.result_dir,'data_user.txt'),'a')
        f.write('ep_list\n')
        f.write(str(ep_list)+'\n')
        f.write('total_reward_list\n')
        f.write(str(total_reward_list)+'\n')
        f.write('ep_loss_list\n')
        f.write(str(ep_loss_list)+'\n')
        f.write('loss_list\n')
        f.write(str(loss_list)+'\n')
        f.write('mae_list\n')
        f.write(str(mae_list)+'\n')
        #f.write(str(weights))
        f.close()
        self.buffer.save_buffer(ep+1)
        #files.download("/content/results/"+str(self.file_name)+"_model.h5")
        #files.download("/content/results/"+str(self.file_name)+"_user.txt")
        #files.download("/content/results/"+str(self.file_name)+"_reward.png")
        #files.download("/content/results/"+str(self.file_name)+"_loss.png")


    def test(self, max_episodes=100):
        self.model.model.load_weights(os.path.join(self.result_dir, str(self.file_name)+'_model.h5'))
        total_test_reward_list=[]
        total_sum_AoI_list=[]
        for ep in range(max_episodes):
            total_test_reward = 0
            done = False
            state = self.env.reset()
            step_count=0
            total_sum_AoI=0
            while not done:
                AoI_list=[int(state[2+1+i]) for i in range(amount_SN)]
                sum_AoI = sum(AoI_list) / amount_SN
                action = self.model.test_get_action(state)
                next_state, reward, done, U_list,_ = self.env.step(action)
                state = next_state
                step_count+=1
                total_test_reward += reward
                total_sum_AoI += sum_AoI
            AoI_list=[int(state[2+1+i]) for i in range(amount_SN)]
            #U_list=[int(state[4+amount_SN+iU]) for iU in range(amount_SN)]
            total_test_reward_list.append(total_test_reward/step_count)
            total_sum_AoI_list.append(total_sum_AoI/step_count)
            print('ep{} AoI/pos={} / {} total_average_AoI={} total_test_reward={} U_list'.format(ep, AoI_list, [state[0+1],state[1+1]], total_sum_AoI/step_count, total_test_reward/step_count,U_list))
        f = open(os.path.join(self.result_dir,'data_user_test.txt'),'a')
        #f.write(str(AoI_list)+'\n')
        f.write('total_episode_reward_list\n')
        f.write(str(total_test_reward_list)+'\n')
        f.write('total_sum_AoI_list\n')
        f.write(str(total_sum_AoI_list)+'\n')
        f.write('average_episodes_test_reward:'+str(sum(total_test_reward_list)/max_episodes)+'\n')
        f.write('average_episodes_sum_AoI:'+str(sum(total_sum_AoI_list)/max_episodes)+'\n')                                               
        f.close()
        print('average_episode_test_reward:'+str(sum(total_test_reward_list)/max_episodes))
        print('average_episode_sum_AoI:'+str(sum(total_sum_AoI_list)/max_episodes))


def main():
    env = AoI()
    agent = Agent(env)
    agent.train(max_episodes=args.train_ep)
    #agent.train(max_episodes=20)
    agent.test(max_episodes=100)

    
import os
from google.colab import files
if __name__ == "__main__":
   main()
#q_network.main_network.save_weights(os.path.join(result_dir, 'model_3.h5'))
