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
#import wandb#ここの環境だけ自分で何とかしましょう！！
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
parser.add_argument('--amount_SN', type=int, default=9)
parser.add_argument('--train_ep', type=int, default=9000)
#parser.add_argument('--train_ep', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.90)
parser.add_argument('--lr', type=float, default=1.0e-5)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.9999)
parser.add_argument('--eps_min', type=float, default=0.10)
args = parser.parse_args([])

print(args.amount_SN)

amount_SN = args.amount_SN
K = 0
file_name = '1_0_e_5'
#file_name = 'test'
result_dir = '/content/drive/MyDrive/previous_env/DQN_for_MARL/epsilon/SN'+str(args.amount_SN)+'/'+str(file_name)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


class AoI(gym.Env): # (1)
    # 定数を定義
    amount_SN = amount_SN#超重要！！
    K = K
    max_AoI = 60
    x_p = 0
    x_m = 1
    y_p = 2
    y_m = 3
        # 初期化
    def __init__(self):
        super(AoI, self).__init__()
        # 初期位置の指定
        self.agent_pos_1 = [0, 0]
        self.agent_pos_2 = [0, 0]
        self.max_AoI = self.max_AoI
        self.amount_SN = self.amount_SN
        self.SNposition_list_base=[[8,7],[8,3],[2,1],[7,0],[4,5],[5,8],[4,0],[0,4],[1,8]]
        self.SNposition_list=[[40*self.SNposition_list_base[i][0],40*self.SNposition_list_base[i][1]] for i in range(self.amount_SN)]
        self.K = K

        self.energy_1 = 1.2*10**5
        self.energy_2 = 1.2*10**5

        # 行動空間と状態空間の型の定義 (2)
        self.action_space = gym.spaces.Discrete(16)
        self.observation_space = gym.spaces.Box(low=0, high=321, shape=(1*2+2*2+self.amount_SN*2,))

    # 環境のリセット (3)
    def reset(self):
        self.E_th = 8000
        self.S_check = 0
        self.done=False
        #ここで新規起動(-1+1)
        self.action_step = 0#self.action_step = -1 + 1 step(-1+1)

        self.agent_pos_1 = [0, 0]#q_1(-1+1)
        self.agent_pos_2 = [0, 0]#q_2(-1+1)

        self.S_list_1=[0 for iS in range(self.amount_SN)]#S_1(-1+1)
        self.S_list_2=[0 for iS in range(self.amount_SN)]#S_2(-1+1)

        self.AoI_list=[1 for iA in range(self.amount_SN)]#A(-1+1)

        self.big_o_list=[0 for iO in range(self.amount_SN)]#O(-1+1)

        self.o_list=[1 for io in range(self.amount_SN)]#o(-1+1)
        
        self.U_list=[1 for iU in range(self.amount_SN)]#U(-1+1)

        self.used_energy_1 = 0
        self.used_energy_2 = 0

        self.residual_energy_1 = self.energy_1 - self.used_energy_1
        self.residual_energy_2 = self.energy_2 - self.used_energy_2
        self.state_list = [self.residual_energy_1/self.energy_1*100,self.residual_energy_2/self.energy_2*100 ,self.agent_pos_1[0], self.agent_pos_1[1],self.agent_pos_2[0],self.agent_pos_2[1]]
        for i in range(self.amount_SN):
          self.state_list.append(self.AoI_list[i])
        for j in range(self.amount_SN):
          self.state_list.append(self.U_list[j])

        self.state = np.array(self.state_list)

        return self.state

      # 環境の1ステップ実行 (3)
    def step(self, action_1,action_2):
        #agent1が情報収集するか問題S(n)
        for iS in range(self.amount_SN):
          if (self.agent_pos_1[0]==self.SNposition_list[iS][0] and self.agent_pos_1[1]==self.SNposition_list[iS][1] and self.U_list[iS] >0):
            self.S_list_1[iS] = 1
          else:
            self.S_list_1[iS] = 0

        #agent2が情報収集するか問題S(n)
        for iS in range(self.amount_SN):
          if (self.agent_pos_2[0]==self.SNposition_list[iS][0] and self.agent_pos_2[1]==self.SNposition_list[iS][1] and self.U_list[iS] >0):
            self.S_list_2[iS] = 1
          else:
            self.S_list_2[iS] = 0

        for iO in range(self.amount_SN):
          self.big_o_list[iO] = min((self.big_o_list[iO]+self.o_list[iO]),1)-max(self.S_list_1[iO],self.S_list_2[iO])#ここはS(n)だよ！not S(n+1)
        
        self.amount_AoI = sum(self.AoI_list) 
        self.reward = (self.amount_AoI) / self.amount_SN
        if self.agent_pos_1[0]==self.agent_pos_2[0] and self.agent_pos_1[1]==self.agent_pos_2[1]:
          self.reward += K
        else:
          pass
        self.action_step += 1
        
        #x_p@x_p=0, x_p@x_m=1, x_p@y_p=2, x_p@y_m=3
        #x_m@x_p=4, x_m@x_m=5, x_m@y_p=6, x_m@y_m=7
        #y_p@x_p=8, y_p@x_m=9, y_p@y_p=10, y_p@y_m=11
        #y_m@x_p=12, y_m@x_m=13, y_m@y_p=14, y_m@y_m=15

        if not self.done:
        #if not done:
            #self.action_step += 1
            #こっから先はn+1の時のこと。ここへきてa(n)をagent(n)に反映
            if action_1 == self.x_p:
                self.agent_pos_1[0] = self.agent_pos_1[0] + 40
            elif action_1 == self.x_m:
                self.agent_pos_1[0] = self.agent_pos_1[0] - 40
            elif action_1 == self.y_p:
                self.agent_pos_1[1] = self.agent_pos_1[1] + 40
            elif action_1 == self.y_m:
                self.agent_pos_1[1] = self.agent_pos_1[1] - 40
            else:
                raise ValueError("Received invalid action={}".format(action_1))
            
            if action_2 == self.x_p:
                self.agent_pos_2[0] = self.agent_pos_2[0] + 40
            elif action_2 == self.x_m:
                self.agent_pos_2[0] = self.agent_pos_2[0] - 40
            elif action_2 == self.y_p:
                self.agent_pos_2[1] = self.agent_pos_2[1] + 40
            elif action_2 == self.y_m:
                self.agent_pos_2[1] = self.agent_pos_2[1] - 40
            else:
                raise ValueError("Received invalid action={}".format(action_2))

                
        #ここでnext_stepのAoIを出力している。
        for iA in range(self.amount_SN):
          if self.S_list_1[iA]==1:
            self.AoI_list[iA]=self.U_list[iA]#ここはA(n+1)=U(n)
          elif self.S_list_2[iA]==1:
            self.AoI_list[iA]=self.U_list[iA]#ここはA(n+1)=U(n)
          else:
            self.AoI_list[iA]=self.AoI_list[iA]+1##ここはA(n+1)=A(n)+1

        for io in range(self.amount_SN):
          if self.action_step % 5==0:# or self.S_list[io]==1:
            self.o_list[io]=1
          else:
            self.o_list[io]=0


        for iU in range(self.amount_SN):
          if (self.big_o_list[iU]==0 and self.o_list[iU]==0):#O(n)=0かつo(n+1)=0
            self.U_list[iU]=0
          elif self.o_list[iU]==1:
            self.U_list[iU]=1
          else:
            self.U_list[iU]= self.U_list[iU] + 1

        ##ここから　パケットドロップ　m　のジャッジ

        #self.m_87 = 1 if self.big_o_87 > 0 and self.o_87==1 else 0
        #self.m_93 = 1 if self.big_o_93 > 0 and self.o_93==1 else 0
        #self.m_21 = 1 if self.big_o_21 > 0 and self.o_21==1 else 0

        for iA in range(self.amount_SN):
          self.AoI_list[iA] = np.clip(self.AoI_list[iA], 0, self.max_AoI )

        #self.double_sensing_check = 0
        #for d_s_c in range(self.amount_SN):
        #  if self.S_list_1[d_s_c]==1 and self.S_list_2[d_s_c]==1:
        #    self.double_sensing_check +=1
        #  else:
        #    pass
        #self.double_sensing_check = int(self.double_sensing_check)
        #self.K = self.K
        #self.amount_AoI = sum(self.AoI_list)
        #self.reward = (self.amount_AoI) / self.amount_SN + self.K*self.double_sensing_check#rewardの裁定、早く書きてぇ(#^^#)(#^^#)
        #self.reward = (self.amount_AoI) / self.amount_SN#rewardの裁定、早く書きてぇ(#^^#)(#^^#)
        #self.S_check = 1 if self.S_87 == 1 or self.S_93 == 1 or self.S_21 == 1 else 0

        #power consumption

        self.used_energy_1 = 810.39097689 + 3.5252811*10**(-6) if self.residual_energy_1 > self.E_th and self.S_check == 1 else 810.39097689
        self.residual_energy_1 = self.residual_energy_1 - self.used_energy_1
        self.used_energy_2 = 810.39097689 + 3.5252811*10**(-6) if self.residual_energy_2 > self.E_th and self.S_check == 1 else 810.39097689
        self.residual_energy_2 = self.residual_energy_2 - self.used_energy_2

        self.check_done_1 = 1 if self.residual_energy_1 < self.E_th else 0
        self.check_done_2 = 1 if self.residual_energy_2 < self.E_th else 0

        self.done = max(self.check_done_1, self.check_done_2) > 0.5

        self.state_list = [self.residual_energy_1/self.energy_1*100,self.residual_energy_2/self.energy_2*100 ,self.agent_pos_1[0], self.agent_pos_1[1],self.agent_pos_2[0],self.agent_pos_2[1]]
        for i in range(self.amount_SN):
          self.state_list.append(self.AoI_list[i])
        for j in range(self.amount_SN):
          self.state_list.append(self.U_list[j])

        self.next_state = np.array(self.state_list)

        return self.next_state, self.reward, self.done, {}

      
def res_actions(x,y):
  res_actions_list = [0,1,2,3]
  if x == 320:
    res_actions_list.remove(0)
  elif x == 0:
    res_actions_list.remove(1)
  else:
    pass
  if y == 320:
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
  for ri in range(len(res_actions_list)):
    res_outputs[0][ri] = outputs[res_actions_list[ri]]
    res_outputs[1][ri] = res_actions_list[ri]
  return res_outputs

def res_next_output_layers(next_outputs,res_next_actions_lists):
  min_output_batch = []
  for ni in range(len(next_outputs)):
    res_next_outputs = []
    for nj in range(len(res_next_actions_lists[ni])):
      res_next_outputs.append(next_outputs[ni][res_next_actions_lists[ni][nj]])
    min_output_batch.append(np.min(res_next_outputs))
  return np.array(min_output_batch)

def choose_res_action(res_outputs):
  argmin_number = np.argmin(res_outputs[0])
  min_action = int(res_outputs[1][argmin_number])
  return min_action


class ReplayBuffer:
    def __init__(self, capacity=40000):
        self.buffer = deque(maxlen=capacity)
    
    def put(self,state, action_1, action_2, reward, next_state, done, res_next_actions_list_1, res_next_actions_list_2):
        self.buffer.append([state, action_1, action_2, reward, next_state, done, res_next_actions_list_1, res_next_actions_list_2])
    
    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, action_1_s, action_2_s, rewards, next_states, done,res_next_actions_list_1_s,res_next_actions_list_2_s = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        res_next_actions_list_1_s = np.array(res_next_actions_list_1_s).reshape(args.batch_size, -1)
        res_next_actions_list_2_s = np.array(res_next_actions_list_2_s).reshape(args.batch_size, -1)
        return states, action_1_s, action_2_s, rewards, next_states, done, res_next_actions_list_1_s,res_next_actions_list_2_s
    
        #states, action_1_s, action_2_s, rewards, next_states, done,res_next_actions_list_1_s,res_next_actions_list_2_s
    def max_sample(self):
        max_sample = self.buffer
        states, action_1_s, action_2_s, rewards, next_states, done,res_next_actions_list_1_s,res_next_actions_list_2_s = map(np.asarray, zip(*max_sample))
        states = np.array(states).reshape(self.size(), -1)
        next_states = np.array(next_states).reshape(self.size(), -1)
        res_next_actions_list_1_s = np.array(res_next_actions_list_1_s).reshape(self.size(), -1)
        res_next_actions_list_2_s = np.array(res_next_actions_list_2_s).reshape(self.size(), -1)        
        return states, action_1_s, action_2_s, rewards, next_states, done, res_next_actions_list_1_s,res_next_actions_list_2_s
      
    def size(self):
        return len(self.buffer)

class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim  = state_dim
        self.action_dim = aciton_dim
        self.epsilon_1 = args.eps
        self.epsilon_2 = args.eps

        self.model_1 = self.create_model()
        self.model_2 = self.create_model()
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

    def predict_on_batch_1(self,state_list_append):##ここ、predictじゃない無い方がええ気がしてるんだが💦💦💦
        #return self.model_1.predict_on_batch(state,state_list)#predict_on_batchではないのかい？？
        return self.model_1.predict_on_batch(state_list_append)#predict_on_batchではないのかい？？    

    def predict_on_batch_2(self,state_list_append):##ここ、predictじゃない無い方がええ気がしてるんだが💦💦💦
        #return self.model_2.predict_on_batch(state,state_list)#predict_on_batchではないのかい？？
        return self.model_2.predict_on_batch(state_list_append)#predict_on_batchではないのかい？？

      
    def get_action_1(self, state):
        state = np.reshape(state, [1, self.state_dim])
        res_actions_list = res_actions(state[0][2],state[0][3])
        self.epsilon_1 *= args.eps_decay
        self.epsilon_1 = max(self.epsilon_1, args.eps_min)
        if np.random.random() < self.epsilon_1:
            return random.choice(res_actions_list)##ここ、restrictで制限しないといけん。この下のリターンも同様じゃ。
        else:
            q_value = self.predict_on_batch_1(state)[0]#まじでここpredictじゃない方がええ気がしてますぅ。        
            res_outputs = res_output_layers(q_value,res_actions_list)
            action = choose_res_action(res_outputs)
            return action#argminへの変更も忘れずに！！]

    def get_action_2(self, state):
        state = np.reshape(state, [1, self.state_dim])
        res_actions_list = res_actions(state[0][4],state[0][5])
        self.epsilon_2 *= args.eps_decay
        self.epsilon_2 = max(self.epsilon_2, args.eps_min)
        if np.random.random() < self.epsilon_2:
            return random.choice(res_actions_list)##ここ、restrictで制限しないといけん。この下のリターンも同様じゃ。
        else:
            q_value = self.predict_on_batch_2(state)[0]#まじでここpredictじゃない方がええ気がしてますぅ。
            res_outputs = res_output_layers(q_value,res_actions_list)
            action = choose_res_action(res_outputs)
            return action#argminへの変更も忘れずに！！]

    def test_get_action_1(self, state):
        state = np.reshape(state, [1, self.state_dim])
        res_actions_list = res_actions(state[0][2],state[0][3])
        q_value = self.predict_on_batch_1(state)[0]#まじでここpredictじゃない方がええ気がしてますぅ。        
        res_outputs = res_output_layers(q_value,res_actions_list)
        action = choose_res_action(res_outputs)
        return action#argminへの変更も忘れずに！！]

    def test_get_action_2(self, state):
        state = np.reshape(state, [1, self.state_dim])
        res_actions_list = res_actions(state[0][4],state[0][5])
        q_value = self.predict_on_batch_2(state)[0]#まじでここpredictじゃない方がええ気がしてますぅ。        
        res_outputs = res_output_layers(q_value,res_actions_list)
        action = choose_res_action(res_outputs)
        return action#argminへの変更も忘れずに！！]

    def train_1(self, states, targets):#ここ、fitじゃなくてもええかもしれんなぁ。
        #self.model.fit(states, targets, batch_size=args.batch_size, epochs=1, verbose=1)
        loss, td_error = self.model_1.train_on_batch(states, targets)
        return loss, td_error

    def train_2(self, states, targets):#ここ、fitじゃなくてもええかもしれんなぁ。
        #self.model.fit(states, targets, batch_size=args.batch_size, epochs=1, verbose=1)
        loss, td_error = self.model_2.train_on_batch(states, targets)
        return loss, td_error


    def evaluate_model_1(self,state_1_lists, targets):
        results_1 = self.model_1.evaluate(state_1_lists, targets,batch_size=len(state_1_lists))
        return results_1

    def evaluate_model_2(self,state_2_lists, targets):
        results_2 = self.model_2.evaluate(state_2_lists, targets,batch_size=len(state_2_lists))
        return results_2

class Agent:
    result_dir=result_dir
    def __init__(self, env):
        self.result_dir = result_dir
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)

        self.target_update_1()
        self.target_update_2()

        self.file_name = file_name

        self.buffer= ReplayBuffer()

        
    def target_update_1(self):
        weights = self.model.model_1.get_weights()
        self.target_model.model_1.set_weights(weights)

        
    def target_update_2(self):
        weights = self.model.model_2.get_weights()
        self.target_model.model_2.set_weights(weights)

    
    def replay_1(self):
        for _ in range(1):
            states, action_1_s, action_2_s, rewards, next_states, done,res_next_actions_list_1_s,res_next_actions_list_2_s = self.buffer.sample()
            targets = self.target_model.predict_on_batch_1(states)
            next_q_values = self.target_model.predict_on_batch_1(next_states)
            future_return = res_next_output_layers(next_q_values,res_next_actions_list_1_s)
            targets[range(args.batch_size), action_1_s] = rewards + (1-done) * future_return * args.gamma
            loss, td_error = self.model.train_1(states, targets)
        return loss, td_error

      
    def replay_2(self):
        for _ in range(1):
            states, action_1_s, action_2_s, rewards, next_states, done,res_next_actions_list_1_s,res_next_actions_list_2_s = self.buffer.sample()
            targets = self.target_model.predict_on_batch_2(states)
            next_q_values = self.target_model.predict_on_batch_2(next_states)
            future_return = res_next_output_layers(next_q_values,res_next_actions_list_2_s)
            targets[range(args.batch_size), action_2_s] = rewards + (1-done) * future_return * args.gamma
            loss, td_error = self.model.train_2(states, targets)
        return loss, td_error


    def evaluate_by_buffer(self):
        states, action_1_s, action_2_s, rewards, next_states, done,res_next_actions_list_1_s,res_next_actions_list_2_s = self.buffer.max_sample()
        targets_1 = self.target_model.predict_on_batch_1(states)
        next_q_values_1 = self.target_model.predict_on_batch_1(next_states)
        future_return = res_next_output_layers(next_q_values_1,res_next_actions_list_1_s)
        targets_1[range(len(future_return)), action_1_s] = rewards + (1-done) * future_return * args.gamma
        results_1 = self.model.evaluate_model_1(states, targets_1)

        targets_2 = self.target_model.predict_on_batch_2(states)
        next_q_values_2 = self.target_model.predict_on_batch_2(next_states)
        future_return = res_next_output_layers(next_q_values_2,res_next_actions_list_2_s)
        targets_2[range(len(future_return)), action_2_s] = rewards + (1-done) * future_return * args.gamma
        results_2 = self.model.evaluate_model_2(states, targets_2)

        return results_1, results_2

      
    def train(self, max_episodes=10000):
        ep_list = []
        ep_loss_list = []
        loss_1_list = []
        loss_2_list = []
        mae_1_list = []
        mae_2_list = []
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
                AoI_list=[int(state[3+3+i]) for i in range(amount_SN)]
                average_AoI = sum(AoI_list) / amount_SN
                action_1 = self.model.get_action_1(state)
                action_2 = self.model.get_action_2(state)
                #return self.next_state, self.reward, self.done, {}                
                next_state, reward, done,  _ = self.env.step(action_1,action_2)
                res_next_actions_list_1=res_actions(next_state[2],next_state[3])
                res_next_actions_list_2=res_actions(next_state[4],next_state[5])
                self.buffer.put(state, action_1, action_2, reward, next_state, done,res_next_actions_list_1, res_next_actions_list_2)#reward×0.01しているん何故？？
                #state, action_1, action_2, reward, next_state, done, res_next_actions_list_1, res_next_actions_list_2
                #ここから下でステップの更新を行っている
                state = next_state
                total_reward += reward
                total_average_AoI += average_AoI
                step_count+=1
                if self.buffer.size() >= args.batch_size:
                    loss_1, td_error_1 = self.replay_1()##ここで２つ分、学習させることができるとよきですね☺☺
                    loss_2, td_error_2 = self.replay_2()##ここで２つ分、学習させることができるとよきですね☺☺
            ep_list.append(ep)    
            reward_list.append(reward)
            total_reward_list.append(total_reward/step_count)
            average_AoI_list.append(average_AoI)
            total_average_AoI_list.append(total_average_AoI/step_count)
            self.target_update_1()
            self.target_update_2()
            if ep > 1:
                AoI_list=[int(state[3+3+i]) for i in range(amount_SN)]
                U_list=[int(state[3+3+amount_SN+i]) for i in range(amount_SN)]
                #U_list=[int(state[2+2+amount_SN+iU]) for iU in range(amount_SN)]
                print('ep{} L1={} M1={} L2={} M2={} F_AoI/U={}/{} taA={} TR={} pos_state={}'\
                      .format(ep, int(loss_1), int(td_error_1), int(loss_2), int(td_error_2),\
                              AoI_list, U_list, \
                              total_average_AoI/step_count ,total_reward/step_count,
                              [state[2],state[3],state[4],state[5]]))
                #print(state[2],state[3],state[4],state[5])
                #if (ep+1) % 100 == 0:
                if (ep+1) % 100 == 0:
                  results_1,results_2 = self.evaluate_by_buffer()
                  ep_loss_list.append(ep)
                  loss_1_list.append(results_1[0])
                  mae_1_list.append(results_1[1])
                  loss_2_list.append(results_2[0])
                  mae_2_list.append(results_2[1])
                  print('loss_1',results_1[0],'mae_1',results_1[1],'loss_2',results_2[0],'mae_2',results_2[1])
                  
                  
        ##ここから保存ゾーン
        #resultsディレクトリを作成
        #result_dir = 'results'
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        fig1 = plt.figure()
        plt.plot(ep_list, total_reward_list);
        plt.show()
        fig1.savefig(os.path.join(self.result_dir,str(self.file_name)+'_reward.png'))
        fig2 = plt.figure()
        plt.plot(ep_loss_list,loss_1_list);
        plt.plot(ep_loss_list,loss_2_list);
        plt.show()
        fig2.savefig(os.path.join(result_dir, str(self.file_name)+'_loss.png'))
        self.target_model.model_1.save_weights(os.path.join(self.result_dir, str(self.file_name)+'_model_1.h5'))
        self.target_model.model_2.save_weights(os.path.join(self.result_dir, str(self.file_name)+'_model_2.h5'))
        f = open(os.path.join(self.result_dir,'data_user.txt'),'a')
        f.write(str(ep_list)+'\n')
        f.write(str(total_reward_list)+'\n')
        f.write(str(ep_loss_list)+'\n')
        f.write(str(loss_1_list)+'\n')
        f.write(str(loss_2_list)+'\n')
        f.write(str(mae_1_list)+'\n')
        f.write(str(mae_2_list)+'\n')
        #f.write(str(weights))
        f.close()
        #files.download("/content/results/"+str(self.file_name)+"_model_1.h5")
        #files.download("/content/results/"+str(self.file_name)+"_model_2.h5")
        #files.download("/content/results/"+str(self.file_name)+"_user.txt")
        #files.download("/content/results/"+str(self.file_name)+"_reward.png")
        #files.download("/content/results/"+str(self.file_name)+"_loss.png")
        ##ここから保存ゾーン

    def test_1(self, max_episodes=10):
        self.model.model_1.load_weights(os.path.join(self.result_dir, str(self.file_name)+"_model_1.h5"))
        self.model.model_2.load_weights(os.path.join(self.result_dir, str(self.file_name)+"_model_2.h5"))
        total_test_reward_list=[]
        total_average_AoI_list=[]
        for ep in range(max_episodes):
            total_test_reward = 0
            total_average_AoI = 0
            done = False
            state = self.env.reset()
            step_count=0
            while not done:
                AoI_list=[int(state[3+3+i]) for i in range(amount_SN)]
                average_AoI = sum(AoI_list) / amount_SN
                action_1 = self.model.test_get_action_1(state)
                action_2 = self.model.test_get_action_2(state)
                next_state, reward, done, _ = self.env.step(action_1,action_2)
                res_next_actions_list_1 = res_actions(next_state[2],next_state[3])
                res_next_actions_list_2 = res_actions(next_state[4],next_state[5])                

                state = next_state
                total_test_reward += reward
                total_average_AoI += average_AoI
                step_count+=1
            total_test_reward_list.append(total_test_reward/step_count)
            total_average_AoI_list.append(total_average_AoI/step_count)            
            AoI_list=[int(state[6+i]) for i in range(amount_SN)]
            U_list=[int(state[6+amount_SN+i]) for i in range(amount_SN)]
            #U_list=[int(state[6+amount_SN+iU]) for iU in range(amount_SN)]
            print('ep{} AoI/U={} / {} state_pos={} total_average_AoI={}'.format(ep, AoI_list, [state[2],state[3],state[4],state[5]], U_list, total_average_AoI/step_count))
        f = open(os.path.join(self.result_dir,'data_user_test.txt'),'a')
        #f.write(str(AoI_list)+'\n')
        f.write('total_episode_reward_list\n')
        f.write(str(total_test_reward_list)+'\n')
        f.write('total_average_AoI_list\n')
        f.write(str(total_average_AoI_list)+'\n')
        f.write('average_episodes_test_reward:'+str(sum(total_test_reward_list)/max_episodes)+'\n')
        f.write('average_episodes_average_AoI:'+str(sum(total_average_AoI_list)/max_episodes)+'\n')                                               
        f.close()
        print('average_episode_test_reward:'+str(sum(total_test_reward_list)/max_episodes))
        print('average_episode_sum_AoI:'+str(sum(total_average_AoI_list)/max_episodes))

def main():
    env = AoI()
    agent = Agent(env)
    agent.train(max_episodes=args.train_ep)
    agent.test_1(max_episodes=10)
    #agent.test_2(max_episodes=0)

if __name__ == "__main__":
   main()
#q_network.main_network.save_weights(os.path.join(result_dir, 'model_3.h5'))