import os
#from google.colab import files
#from google.colab import drive
#drive.mount('/content/drive')
#import wandb#ここの環境だけ自分で何とかしましょう！！
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
import time
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float32')


parser = argparse.ArgumentParser()
parser.add_argument('--amount_SN', type=int, default=10)
parser.add_argument('--train_ep', type=int, default=9000)
#parser.add_argument('--train_ep', type=int, default=30)
parser.add_argument('--gamma', type=float, default=0.90)
parser.add_argument('--lr', type=float, default=5.0e-4)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.9999)
parser.add_argument('--eps_min', type=float, default=0.10)
args = parser.parse_args([])

print(args.amount_SN)

amount_SN = args.amount_SN
#file_name = 'again_3times_same_start_Nposi_K_00_5_1_0e_5_SN9_Multi'
file_name = '5_0_e_5_timeslot'
#file_name = 'SN'+str(amount_SN)+'_1'+

#result_dir = '/content/drive/MyDrive/new_env/MADQN/9000_3times/'+str(file_name)
result_dir = '/home/daisuke/google_drive/modify_env/MADQN/SN10'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

    
class AoI(gym.Env): # (1)
    # 定数を定義
    amount_SN = amount_SN#超重要！！
    max_AoI = 100#??time_slot分？
    x_p__x_p = 0
    x_p__x_m = 1
    x_p__y_p = 2
    x_p__y_m = 3
    x_m__x_p = 4
    x_m__x_m = 5
    x_m__y_p = 6
    x_m__y_m = 7
    y_p__x_p = 8
    y_p__x_m = 9
    y_p__y_p = 10
    y_p__y_m = 11
    y_m__x_p = 12
    y_m__x_m = 13
    y_m__y_p = 14
    y_m__y_m = 15

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
        self.action_space = gym.spaces.Discrete(16)
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
    def step(self, action):
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

        if action == self.x_p__x_p:
            self.agent_pos_1[0] = self.agent_pos_1[0] + 50
            self.agent_pos_2[0] = self.agent_pos_2[0] + 50
        elif action == self.x_p__x_m:
            self.agent_pos_1[0] = self.agent_pos_1[0] + 50
            self.agent_pos_2[0] = self.agent_pos_2[0] - 50
        elif action == self.x_p__y_p:
            self.agent_pos_1[0] = self.agent_pos_1[0] + 50
            self.agent_pos_2[1] = self.agent_pos_2[1] + 50
        elif action == self.x_p__y_m:
            self.agent_pos_1[0] = self.agent_pos_1[0] + 50
            self.agent_pos_2[1] = self.agent_pos_2[1] - 50
        #切り替え
        elif action == self.x_m__x_p:
            self.agent_pos_1[0] = self.agent_pos_1[0] - 50
            self.agent_pos_2[0] = self.agent_pos_2[0] + 50
        elif action == self.x_m__x_m:
            self.agent_pos_1[0] = self.agent_pos_1[0] - 50
            self.agent_pos_2[0] = self.agent_pos_2[0] - 50
        elif action == self.x_m__y_p:
            self.agent_pos_1[0] = self.agent_pos_1[0] - 50
            self.agent_pos_2[1] = self.agent_pos_2[1] + 50
        elif action == self.x_m__y_m:
            self.agent_pos_1[0] = self.agent_pos_1[0] - 50
            self.agent_pos_2[1] = self.agent_pos_2[1] - 50
        #切り替え
        elif action == self.y_p__x_p:
            self.agent_pos_1[1] = self.agent_pos_1[1] + 50
            self.agent_pos_2[0] = self.agent_pos_2[0] + 50
        elif action == self.y_p__x_m:
            self.agent_pos_1[1] = self.agent_pos_1[1] + 50
            self.agent_pos_2[0] = self.agent_pos_2[0] - 50
        elif action == self.y_p__y_p:
            self.agent_pos_1[1] = self.agent_pos_1[1] + 50
            self.agent_pos_2[1] = self.agent_pos_2[1] + 50
        elif action == self.y_p__y_m:
            self.agent_pos_1[1] = self.agent_pos_1[1] + 50
            self.agent_pos_2[1] = self.agent_pos_2[1] - 50
        #切り替え
        elif action == self.y_m__x_p:
            self.agent_pos_1[1] = self.agent_pos_1[1] - 50
            self.agent_pos_2[0] = self.agent_pos_2[0] + 50
        elif action == self.y_m__x_m:
            self.agent_pos_1[1] = self.agent_pos_1[1] - 50
            self.agent_pos_2[0] = self.agent_pos_2[0] - 50
        elif action == self.y_m__y_p:
            self.agent_pos_1[1] = self.agent_pos_1[1] - 50
            self.agent_pos_2[1] = self.agent_pos_2[1] + 50
        elif action == self.y_m__y_m:
            self.agent_pos_1[1] = self.agent_pos_1[1] - 50
            self.agent_pos_2[1] = self.agent_pos_2[1] - 50
        else:
            raise ValueError("Received invalid action={}".format(action))

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
    
def res_actions(x_1,y_1,x_2,y_2):
  res_actions_list_1 = [0,1,2,3]
  res_actions_list_2 = [0,1,2,3]
  res_actions_list =[]

  #Agent_1の行動制限
  if x_1 == 400:
    res_actions_list_1.remove(0)
  elif x_1 == 0:
    res_actions_list_1.remove(1)
  else:
    pass
  if y_1 == 400:
    res_actions_list_1.remove(2)
  elif y_1 == 0:
    res_actions_list_1.remove(3)
  else:
    pass
    
  #Agent_2の行動制限
  if x_2 == 400:
    res_actions_list_2.remove(0)
  elif x_2 == 0:
    res_actions_list_2.remove(1)
  else:
    pass
  if y_2 == 400:
    res_actions_list_2.remove(2)
  elif y_2 == 0:
    res_actions_list_2.remove(3)
  else:
    pass

  for i_1 in range(len(res_actions_list_1)):
    for i_2 in range(len(res_actions_list_2)):
      res_actions_list.append(4*res_actions_list_1[i_1]+res_actions_list_2[i_2])

  res_actions_list = np.array(res_actions_list)
  return res_actions_list

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

    def save_buffer(self):
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        f = open(os.path.join(self.result_dir,'check_point_data_user.txt'),'a')
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

    def predict_on_batch(self,state):##ここ、predictじゃない無い方がええ気がしてるんだが💦💦💦
        return self.model.predict_on_batch(state)#predict_on_batchではないのかい？？

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        res_actions_list = res_actions(state[0][0+1],state[0][1+1],state[0][2+1],state[0][3+1])
        q_value = self.predict_on_batch(state)[0]#まじでここpredictじゃない方がええ気がしてますぅ。
        if np.random.random() < self.epsilon:
            return random.choice(res_actions_list)##ここ、restrictで制限しないといけん。この下のリターンも同様じゃ。
        else:
            res_outputs = res_output_layers(q_value,res_actions_list)
            action = choose_res_action(res_outputs)
            return action#argminへの変更も忘れずに！！]

    def test_get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        res_actions_list = res_actions(state[0][0+1],state[0][1+1],state[0][2+1],state[0][3+1])
        q_value = self.predict_on_batch(state)[0]#まじでここpredictじゃない方がええ気がしてますぅ。        
        res_outputs = res_output_layers(q_value,res_actions_list)
        action = choose_res_action(res_outputs)
        return action#argminへの変更も忘れずに！！

    def train(self, states, targets):#ここ、fitじゃなくてもええかもしれんなぁ。
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
        for _ in range(3):
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
        #q_preds = targets#これは重みを同期したあとにやっているから問題ない！
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
        sum_AoI_list = []
        total_sum_AoI_list = []
        for ep in range(max_episodes):
            total_reward = 0
            total_sum_AoI = 0
            done = False
            state = self.env.reset()
            step_count=0
            while not done:
                AoI_list=[int(state[2+2+1+i]) for i in range(amount_SN)]
                sum_AoI = sum(AoI_list) / amount_SN
                action = self.model.get_action(state)
                next_state, reward, done, U_list, _ = self.env.step(action)
                res_next_actions_list = res_actions(next_state[0+1],next_state[1+1],next_state[2+1],next_state[3+1])
                self.buffer.put(state, action, reward, next_state, done,res_next_actions_list)#reward×0.01しているん何故？？
                state = next_state
                total_reward += reward
                total_sum_AoI += sum_AoI
                step_count+=1
                if self.buffer.size() >= args.batch_size:
                    loss, td_error = self.replay()
            self.target_update()
            ep_list.append(ep)
            reward_list.append(reward)#こいつらがあんまり意味ない
            total_reward_list.append(total_reward/step_count)
            sum_AoI_list.append(sum_AoI)#こいつらがあんまり意味ない
            total_sum_AoI_list.append(total_sum_AoI/step_count)
            if ep > 0:
                AoI_list=[int(state[2+2+1+i]) for i in range(amount_SN)]
                #U_list=[int(state[2+2+amount_SN+iU]) for iU in range(amount_SN)]
                print('ep{} loss={} mae={} F_AoI/pos={}/{} taA={} TR={}  aA={} R={} U_list={}'\
                      .format(ep, loss, td_error,\
                              AoI_list, [state[0+1],state[1+1],state[2+1],state[3+1]], \
                              total_sum_AoI/step_count ,total_reward/step_count,
                              sum_AoI, reward, U_list))
                if (ep+1) % 100 == 0:
                  results = self.evaluate_by_buffer()
                  ep_loss_list.append(ep)
                  loss_list.append(results[0])
                  mae_list.append(results[1])
                  print('loss',results[0],'mae',results[1])
                  
                if (ep+1) % 5000 == 0:
                  self.target_model.model.save_weights(os.path.join(self.result_dir, str(self.file_name)+'check_point_5000_model.h5'))
                  self.buffer.save_buffer()#reward×0.01しているん何故？？                                      
                  f = open(os.path.join(self.result_dir,'check_point_data_user.txt'),'a')
                  f.write(str(args.amount_SN)+'\n')
                  f.write(str(args.train_ep)+'\n')
                  f.write(str(args.gamma)+'\n')
                  f.write(str(args.lr)+'\n')
                  f.write(str(args.batch_size)+'\n')
                  f.write(str(args.eps)+'\n')
                  f.write(str(args.eps_decay)+'\n')
                  f.write(str(args.eps_min)+'\n')
                  f.write(str(ep_list)+'\n')
                  f.write('total_reward_list\n')
                  f.write(str(total_reward_list)+'\n')
                  f.write('total_sum_AoI_list\n')
                  f.write(str(total_sum_AoI_list)+'\n')
                  f.write(str(ep_loss_list)+'\n')
                  f.write(str(loss_list)+'\n')
                  f.write(str(mae_list)+'\n')
                  f.close()
                  print("Now we save model")
                
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
        f.write(str(args.amount_SN)+'\n')
        f.write(str(args.train_ep)+'\n')
        f.write(str(args.gamma)+'\n')
        f.write(str(args.lr)+'\n')
        f.write(str(args.batch_size)+'\n')
        f.write(str(args.eps)+'\n')
        f.write(str(args.eps_decay)+'\n')
        f.write(str(args.eps_min)+'\n')
        f.write(str(ep_list)+'\n')
        f.write('total_reward_list\n')
        f.write(str(total_reward_list)+'\n')
        f.write('total_sum_AoI_list\n')
        f.write(str(total_sum_AoI_list)+'\n')
        f.write(str(ep_loss_list)+'\n')
        f.write(str(loss_list)+'\n')
        f.write(str(mae_list)+'\n')
        f.close()

    def test(self, max_episodes=100):
        self.model.model.load_weights(os.path.join(self.result_dir, str(self.file_name)+'_model.h5'))
        total_test_reward_list=[]
        total_sum_AoI_list=[]
        total_latency_list=[]
        for ep in range(max_episodes):
            latency_list=[]
            total_test_reward = 0
            done = False
            state = self.env.reset()
            step_count=0
            total_sum_AoI=0
            while not done:
                AoI_list=[int(state[2+2+1+i]) for i in range(amount_SN)]
                sum_AoI = sum(AoI_list) / amount_SN
                start = time.time()
                action = self.model.test_get_action(state)
                latency = time.time() - start
                next_state, reward, done, U_list,_ = self.env.step(action)
                state = next_state
                step_count+=1
                total_test_reward += reward
                total_sum_AoI += sum_AoI
                latency_list.append(latency)
            AoI_list=[int(state[4+1+i]) for i in range(amount_SN)]
            #U_list=[int(state[4+amount_SN+iU]) for iU in range(amount_SN)]
            total_test_reward_list.append(total_test_reward/step_count)
            total_sum_AoI_list.append(total_sum_AoI/step_count)
            print('ep{} AoI/pos={} / {} total_average_AoI={} total_test_reward={} U_list'.format(ep, AoI_list, [state[0+1],state[1+1],state[2+1],state[3+1]], total_sum_AoI/step_count, total_test_reward/step_count,U_list))
            print()
            print(latency_list)
            total_latency_list.append(latency_list)
            #latency_list.pop(0)
            #print(max(latency_list))
        f = open(os.path.join(self.result_dir,'data_user_test.txt'),'a')
        #f.write(str(AoI_list)+'\n')
        f.write('total_latency_list\n')
        f.write(str(total_latency_list))
        #f.write(str(max(total_latency_list)))
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
    env = AoI()#ここを私のに変えましょうぜい✊✊✊
    agent = Agent(env)
    #agent.train(max_episodes=args.train_ep)
    #agent.train(max_episodes=21)
    agent.test(max_episodes=10)

#lossの出現割る数、直そうね。
if __name__ == "__main__":
   main()
#q_network.main_network.save_weights(os.path.join(result_dir, 'model_3.h5'))
