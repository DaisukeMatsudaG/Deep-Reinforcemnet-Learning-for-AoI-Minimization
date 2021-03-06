import os
#from google.colab import files
#from google.colab import drive
#drive.mount('/content/drive')
#import wandb#ここの環境だけ自分で何とかしましょう！！
#!pip install tensorflow==2.5.0
import tensorflow as tf
print('TensorFlow', tf.__version__)
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import gym
import os
import argparse
import numpy as np
from collections import deque
import random
#simport matplotlib.pyplot as plt
import time

tf.keras.backend.set_floatx('float32')
#wandb.init(name='DQN', project="deep-rl-tf2")

parser = argparse.ArgumentParser()
parser.add_argument('--amount_SN', type=int, default=10)
parser.add_argument('--train_ep', type=int, default=9000)
#parser.add_argument('--train_ep', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.90)
parser.add_argument('--lr', type=float, default=5.0e-4)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.9999)
parser.add_argument('--eps_min', type=float, default=0.10)
args = parser.parse_args([])

print(args.amount_SN)

amount_SN = args.amount_SN#超重要！この行の３行後も変えなさい！！
file_name = '1_0_e_4_timeslot'
#file_name = 'SN'+str(amount_SN)+'_1'+

#result_dir = '/content/drive/MyDrive/new_env/MADQN/9000_3times/'+str(file_name)
result_dir = '/home/daisuke/google_drive/modify_env/R_MADQN/SN10'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


class AoI(gym.Env): # (1)
    # 定数を定義
    amount_SN = amount_SN#超重要！！
    max_AoI = 100
    x_p = 0
    x_m = 1
    y_p = 2
    y_m = 3

        #初期化
    def __init__(self):
        super(AoI, self).__init__()
        # 初期位置の指定
        self.agent_pos_1 = [200, 200]
        self.agent_pos_2 = [200, 200]
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
        ####ここで指数分布の確率を出力する
        x =  np.arange(0, 100, 1)
        y1= [exp_dist(2/1,i) for i in x]
        y2= [exp_dist(2/10,i) for i in x]
        y3= [exp_dist(2/20,i) for i in x]
        y4= [exp_dist(2/40,i) for i in x]
        self.exponential_distribution_list=[y1,y2,y3,y4]
        ####ここで指数分布の確率を出力する

        # 行動空間と状態空間の型の定義 (2)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=321, shape=(1+2*2+self.amount_SN,))

    # 環境のリセット (3)
    def reset(self):
        self.time_slot=0

        self.agent_pos_1 = [200, 200]#q_1(-1+1)
        self.agent_pos_2 = [200, 200]#q_2(-1+1)

        self.S_list_1=[0 for iS in range(self.amount_SN)]#S_1(-1+1)
        self.S_list_2=[0 for iS in range(self.amount_SN)]#S_2(-1+1)

        self.AoI_list=[1 for iA in range(self.amount_SN)]#A(-1+1)

        self.big_o_list=[0 for iO in range(self.amount_SN)]#O(-1+1)

        self.o_list=[1 for io in range(self.amount_SN)]#o(-1+1)
        self.previous_time_slot_list=[0 for io in range(self.amount_SN)]
        
        self.U_list=[1 for iU in range(self.amount_SN)]#U(-1+1)

        self.state_list = [self.time_slot,self.agent_pos_1[0], self.agent_pos_1[1],self.agent_pos_2[0],self.agent_pos_2[1]]
        for i in range(self.amount_SN):
          self.state_list.append(self.AoI_list[i])

        self.state = np.array(self.state_list)

        return self.state

      # 環境の1ステップ実行 (3)
    def step(self, action_1, action_2):
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
        self.reward = self.amount_AoI / self.amount_SN


        self.time_slot += 1##ここですここ！！マスが切り替わるところです！！
        #こっから先はn+1の時のこと。ここへきてa(n)をagent(n)に反映

        if action_1 == self.x_p:
            self.agent_pos_1[0] = self.agent_pos_1[0] + 50
        elif action_1 == self.x_m:
            self.agent_pos_1[0] = self.agent_pos_1[0] - 50
        elif action_1 == self.y_p:
            self.agent_pos_1[1] = self.agent_pos_1[1] + 50
        elif action_1 == self.y_m:
            self.agent_pos_1[1] = self.agent_pos_1[1] - 50
        else:
            raise ValueError("Received invalid action={}".format(action_1))
        
        if action_2 == self.x_p:
            self.agent_pos_2[0] = self.agent_pos_2[0] + 50
        elif action_2 == self.x_m:
            self.agent_pos_2[0] = self.agent_pos_2[0] - 50
        elif action_2 == self.y_p:
            self.agent_pos_2[1] = self.agent_pos_2[1] + 50
        elif action_2 == self.y_m:
            self.agent_pos_2[1] = self.agent_pos_2[1] - 50
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
        
        ##ここで、指数分布に従ってデータ収集を行うか決定する。
        #IoTデバイスごとに確率分布
          #乱数を生成する
          #指数分布の数字を引っ張ってくる
          #if文で比較しましょう！タイムスロットを記録しましょう！
        for io in range(self.amount_SN):
          #確率値
          self.probability_for_exponential_distribution=np.random.random()
          #確率分布のリスト[パラメータy1~y4][参照したい時間間隔]
          #[参照したい時間間隔]=[今のタイムスロット ー previousのタイムスロット]
          #確率分布のリスト[パラメータy1~y4][今のタイムスロット ー previousのタイムスロット]
          #条件を満たしたとき、if文後、o_listの表示
          self.threshold_probability = self.exponential_distribution_list[self.SN_distribution_list[io]-1][self.time_slot-self.previous_time_slot_list[io]]
          if self.probability_for_exponential_distribution < self.threshold_probability:
            self.o_list[io]=1
            self.previous_time_slot_list[io]=self.time_slot
            #timeslotに記録しなくちゃいかん！
          else:
            self.o_list[io]=0

        #for io in range(self.amount_SN):
        #  if self.action_step % 5==0:# or self.S_list[io]==1:#10ステップごとに情報をリフレッシュ＋先ほどUAVが回収なら収集！
        #    self.o_list[io]=1
        #  else:
        #    self.o_list[io]=0

        for iU in range(self.amount_SN):
          if (self.big_o_list[iU]==0 and self.o_list[iU]==0):#O(n)=0かつo(n+1)=0
            self.U_list[iU]=0
          elif self.o_list[iU]==1:
            self.U_list[iU]=1
          else:
            self.U_list[iU]= self.U_list[iU] + 1

        for iA in range(self.amount_SN):
          self.AoI_list[iA] = np.clip(self.AoI_list[iA], 0, self.max_AoI )

        self.done = self.time_slot == 100##100回リワードを出力したことになる！

        self.state_list=[self.time_slot,self.agent_pos_1[0],self.agent_pos_1[1],self.agent_pos_2[0],self.agent_pos_2[1]]

        for i in range(self.amount_SN):
          self.state_list.append(self.AoI_list[i])

        self.next_state = np.array(self.state_list)

        return self.next_state, self.reward, self.done,self.U_list, {}

def exp_dist(lambda_, x):

    return 1-np.exp(- lambda_*x)

      
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
        self.hidden_layer_model_1 = Model(inputs=self.model_1.input,outputs=self.model_1.get_layer('Layer_0').output)
        self.hidden_layer_model_2 = Model(inputs=self.model_2.input,outputs=self.model_2.get_layer('Layer_0').output)


    def create_model(self):
        model = tf.keras.Sequential([
            InputLayer(((self.state_dim - 2)*5+self.state_dim*1,)),
            Dense(self.state_dim*1, activation='relu',name='Layer_0'),
            Dense(200, activation='relu'),
            Dense(200, activation='relu'),
            Dense(200, activation='relu'),
            Dense(self.action_dim)
        ])
        model.compile(loss='mse',metrics='mae', optimizer=Adam(args.lr))
        return model

    def hidden_1_outputs(self,return_state,state_1_list):
        return_state = np.reshape(return_state, [1, self.state_dim*1])
        state_1_list =np.reshape(state_1_list, [1, (self.state_dim-2)*5])
        state_1_list_append = np.append(state_1_list,return_state)
        state_1_list_append = np.reshape(state_1_list_append, [1, len(state_1_list_append)])
        return self.hidden_layer_model_1.predict_on_batch(state_1_list_append) 

    def hidden_2_outputs(self,return_state,state_2_list):
        return_state = np.reshape(return_state, [1, self.state_dim*1])
        state_2_list =np.reshape(state_2_list, [1, (self.state_dim-2)*5])
        state_2_list_append = np.append(state_2_list,return_state)
        state_2_list_append = np.reshape(state_2_list_append, [1, len(state_2_list_append)])
        return self.hidden_layer_model_1.predict_on_batch(state_2_list_append) 

    def predict_on_batch_1(self,state_list_append):##ここ、predictじゃない無い方がええ気がしてるんだが💦💦💦
        #return self.model_1.predict_on_batch(state,state_list)#predict_on_batchではないのかい？？
        return self.model_1.predict_on_batch(state_list_append)#predict_on_batchではないのかい？？
    
    def predict_on_batch_2(self,state_list_append):##ここ、predictじゃない無い方がええ気がしてるんだが💦💦💦
        #return self.model_2.predict_on_batch(state,state_list)#predict_on_batchではないのかい？？
        return self.model_2.predict_on_batch(state_list_append)#predict_on_batchではないのかい？？
    
    def get_action_1(self, return_state, state_list):
        return_state = np.reshape(return_state, [1, self.state_dim*1])
        res_actions_list = res_actions(state_list[4][1],state_list[4][2])
        state_list =np.reshape(state_list, [1, (self.state_dim-2)*5])
        state_list_append = np.append(state_list,return_state)
        state_list_append = np.reshape(state_list_append, [1, len(state_list_append)])
        self.epsilon_1 *= args.eps_decay
        self.epsilon_1 = max(self.epsilon_1, args.eps_min)
        if np.random.random() < self.epsilon_1:
            return random.choice(res_actions_list)##ここ、restrictで制限しないといけん。この下のリターンも同様じゃ。
        else:
            q_value = self.predict_on_batch_1(state_list_append)[0]#まじでここpredictじゃない方がええ気がしてますぅ。
            res_outputs = res_output_layers(q_value,res_actions_list)
            action = choose_res_action(res_outputs)
            return action#argminへの変更も忘れずに！！]

    def get_action_2(self, return_state, state_list):
        return_state = np.reshape(return_state, [1, self.state_dim*1])
        res_actions_list = res_actions(state_list[4][1],state_list[4][2])
        state_list =np.reshape(state_list, [1, (self.state_dim-2)*5])
        state_list_append = np.append(state_list,return_state)
        state_list_append = np.reshape(state_list_append, [1, len(state_list_append)])
        self.epsilon_2 *= args.eps_decay
        self.epsilon_2 = max(self.epsilon_2, args.eps_min)
        if np.random.random() < self.epsilon_2:
            return random.choice(res_actions_list)##ここ、restrictで制限しないといけん。この下のリターンも同様じゃ。
        else:
            q_value = self.predict_on_batch_2(state_list_append)[0]#まじでここpredictじゃない方がええ気がしてますぅ。
            res_outputs = res_output_layers(q_value,res_actions_list)
            action = choose_res_action(res_outputs)
            return action#argminへの変更も忘れずに！！]

    def test_get_action_1(self, return_state, state_list):
        return_state = np.reshape(return_state, [1, self.state_dim*1])
        res_actions_list = res_actions(state_list[4][1],state_list[4][2])
        state_list =np.reshape(state_list, [1, (self.state_dim-2)*5])
        state_list_append = np.append(state_list,return_state)
        state_list_append = np.reshape(state_list_append, [1, len(state_list_append)])
        q_value = self.predict_on_batch_1(state_list_append)[0]#まじでここpredictじゃない方がええ気がしてますぅ。
        res_outputs = res_output_layers(q_value,res_actions_list)
        action = choose_res_action(res_outputs)
        return action#argminへの変更も忘れずに！！]

    def test_get_action_2(self, return_state, state_list):
        return_state = np.reshape(return_state, [1, self.state_dim*1])
        res_actions_list = res_actions(state_list[4][1],state_list[4][2])
        state_list =np.reshape(state_list, [1, (self.state_dim-2)*5])
        state_list_append = np.append(state_list,return_state)
        state_list_append = np.reshape(state_list_append, [1, len(state_list_append)])
        q_value = self.predict_on_batch_2(state_list_append)[0]#まじでここpredictじゃない方がええ気がしてますぅ。
        res_outputs = res_output_layers(q_value,res_actions_list)
        action = choose_res_action(res_outputs)
        return action#argminへの変更も忘れずに！！]


class Agent:
    result_dir=result_dir
    def __init__(self, env):
        self.result_dir = result_dir
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)

        #self.target_update_1()
        #self.target_update_2()

        self.file_name = file_name

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
                AoI_list=[int(state[1+2*2+i]) for i in range(amount_SN)]
                average_AoI = sum(AoI_list) / amount_SN
                action_1 = self.model.get_action_1(state)
                action_2 = self.model.get_action_2(state)
                #return self.next_state, self.reward, self.done, {}                
                next_state, reward, done,  _, U_list = self.env.step(action_1,action_2)
                res_next_actions_list_1=res_actions(next_state[1],next_state[2])
                res_next_actions_list_2=res_actions(next_state[3],next_state[4])
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
                AoI_list=[int(state[1+2*2+i]) for i in range(amount_SN)]
                #U_list=[int(state[2+2+amount_SN+iU]) for iU in range(amount_SN)]
                print('ep{} L1={} M1={} L2={} M2={} F_AoI/U={}/{} taA={} TR={} pos_state={}'\
                      .format(ep, int(loss_1), int(td_error_1), int(loss_2), int(td_error_2),\
                              AoI_list, U_list, \
                              total_average_AoI/step_count ,total_reward/step_count,
                              [state[1],state[2],state[3],state[4]]))
                #print(state[2],state[3],state[4],state[5])
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
        total_latency_list_1 = []
        total_latency_list_2 = []
        for ep in range(max_episodes):
            latency_list_1 = []
            latency_list_2 = []
            total_reward = 0
            total_average_AoI = 0
            done = False
            state = self.env.reset()
            #before_state=state
            state_1 = np.delete(state,[3,4]) 
            state_2 = np.delete(state,[1,2])
            step_count=0
            state_1_list=np.array([state_1, state_1, state_1, state_1, state_1])
            next_state_1_list=state_1_list
            state_2_list=np.array([state_2, state_2, state_2, state_2, state_2])
            next_state_2_list=state_2_list
            return_state_1_before=state
            return_state_2_before=state
            while not done:
                AoI_list=[int(state[1+2+2+i]) for i in range(amount_SN)]
                average_AoI = sum(AoI_list) / amount_SN
                start_1 = time.time()
                action_1 = self.model.test_get_action_1(return_state_1_before,state_1_list)
                latency_1 = time.time() - start_1
                latency_list_1.append(latency_1)
                action_2 = self.model.test_get_action_2(return_state_2_before,state_2_list)
                next_state, reward, done, U_list, _ = self.env.step(action_1,action_2)
                next_state_1 = np.delete(next_state,[3,4]) 
                next_state_2 = np.delete(next_state,[1,2])                
                next_state_1_list = np.vstack((state_1_list,next_state_1))
                next_state_1_list = np.delete(next_state_1_list,0,0)
                next_state_2_list = np.vstack((state_2_list,next_state_2))
                next_state_2_list = np.delete(next_state_2_list,0,0)
                start_2 = time.time()
                return_state_1_after = self.model.hidden_1_outputs(return_state_1_before,state_1_list)
                latency_2 = time.time() - start_2
                latency_list_2.append(latency_2)
                return_state_2_after = self.model.hidden_2_outputs(return_state_2_before,state_2_list)                
                return_state_1_before = return_state_1_after
                return_state_2_before = return_state_2_after

                state = next_state
                state_1_list = next_state_1_list
                state_2_list = next_state_2_list
                total_reward += reward
                total_average_AoI += average_AoI
                step_count+=1

            total_test_reward_list.append(total_reward/step_count)
            total_average_AoI_list.append(total_average_AoI/step_count)            
            AoI_list=[int(state[1+4+i]) for i in range(amount_SN)]
            total_latency_list_1.append(latency_list_1)
            total_latency_list_2.append(latency_list_2)
            print(AoI_list)
        f = open(os.path.join(self.result_dir,'data_user_test.txt'),'a')
        f.write('total_latency_list_1\n')
        f.write(str(total_latency_list_1)+'\n')

        f.write('total_latency_list_2\n')
        f.write(str(total_latency_list_2)+'\n')

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
    env = AoI()#ここを私のに変えましょうぜい✊✊✊
    agent = Agent(env)
    #agent.train(max_episodes=args.train_ep)
    agent.test_1(max_episodes=10)
    #agent.test_2(max_episodes=0)
#lossの出現割る数、直そうね。
if __name__ == "__main__":
   main()
#q_network.main_network.save_weights(os.path.join(result_dir, 'model_3.h5'))
