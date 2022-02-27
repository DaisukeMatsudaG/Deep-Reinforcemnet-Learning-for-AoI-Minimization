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
import os
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
import statistics

tf.keras.backend.set_floatx('float32')

parser = argparse.ArgumentParser()
parser.add_argument('--amount_SN', type=int, default=8)
parser.add_argument('--train_ep', type=int, default=9000)
parser.add_argument('--gamma', type=float, default=0.90)
parser.add_argument('--lr', type=float, default=5.0e-4)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.9999)
parser.add_argument('--eps_min', type=float, default=0.10)
parser.add_argument('--threshold', type=float, default=0.10)
parser.add_argument('--duration_threshold', type=float, default=5)
args = parser.parse_args([])

print(args.amount_SN)

amount_SN = args.amount_SN#超重要！この行の３行後も変えなさい！！

#DQN_for_MARL
file_name_1 = '5_0_e_4_timeslot'
#no_DQN_for_MARL#1_0_e_3か!!
file_name_2 = '1_0_e_3_timeslot'
file_name = 'change_1000_'+str(int(100*args.threshold))+'_'+str(args.duration_threshold)+'_first_and_first'

result_dir_1 = '/content/drive/MyDrive/modi_env/DQN_for_MARL/epsilon/SN'+str(args.amount_SN)+'/'+str(file_name_1)
result_dir_2 = '/content/drive/MyDrive/modi_env/no_DQN_for_MARL/epsilon/SN'+str(args.amount_SN)+'/'+str(file_name_2)
result_dir = '/content/drive/MyDrive/modi_env/total_system/SN'+str(args.amount_SN)+'/DQN_for_MARL_and_No_DQN_for_MARL/'+str(file_name)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

class AoI(gym.Env): # (1)
    amount_SN = amount_SN#超重要！！
    max_AoI = 100
    x_p = 0
    x_m = 1
    y_p = 2
    y_m = 3
    result_dir_2 = result_dir_2
    file_name_2 = file_name_2

    def __init__(self):
        self.result_dir_2 = result_dir_2
        self.file_name_2 = file_name_2
        self.model = ActionStateModel(1+2*2+amount_SN, 4)
        self.model.model_3.load_weights(os.path.join(self.result_dir_2, str(self.file_name_2)+"_model_1.h5"))
        self.model.model_4.load_weights(os.path.join(self.result_dir_2, str(self.file_name_2)+"_model_2.h5"))
        
        super(AoI, self).__init__()
        self.agent_pos_1 = [200, 200]
        self.agent_pos_2 = [200, 200]
        self.max_AoI = self.max_AoI
        self.amount_SN = self.amount_SN
        self.SNposition_list_base=[[8,4],[4,0],[0,0],[8,0],[0,4],[4,8],[8,8],[0,8],[2,6],[6,2],[6,6],[2,2]]
        self.SNposition_list=[[50*self.SNposition_list_base[i][0],50*self.SNposition_list_base[i][1]] for i in range(self.amount_SN)]
        self.SN_distribution_list=[1,2,3,4,1,2,3,4,1,2,3,4]
        self.time_slot=0
        self.done=False
        x = np.arange(0, 100, 1)
        y1= [exp_dist(2/1,i) for i in x]
        y2= [exp_dist(2/10,i) for i in x]
        y3= [exp_dist(2/20,i) for i in x]
        y4= [exp_dist(2/40,i) for i in x]
        self.exponential_distribution_list=[y1,y2,y3,y4]
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=321, shape=(1+2*2+self.amount_SN,))        
        
    def reset(self):
        self.time_slot=0
        self.agent_pos_1 = [200, 200]#q_1(-1+1)
        self.agent_pos_2 = [200, 200]#q_2(-1+1)

        self.S_list_1=[0 for iS in range(self.amount_SN)]#S_1(-1+1)
        self.S_list_2=[0 for iS in range(self.amount_SN)]#S_2(-1+1)

        self.AoI_list=[1 for iA in range(self.amount_SN)]#A(-1+1)
        self.AoI_list_UAV_1=[1 for iA in range(self.amount_SN)]
        self.AoI_list_UAV_2=[1 for iA in range(self.amount_SN)]
        self.D_1_list=[]
        self.D_2_list=[]

        self.big_o_list=[0 for iO in range(self.amount_SN)]#O(-1+1)

        self.o_list=[1 for io in range(self.amount_SN)]#o(-1+1)
        self.previous_time_slot_list=[0 for io in range(self.amount_SN)]
        
        self.U_list=[1 for iU in range(self.amount_SN)]#U(-1+1)
        self.U_list_UAV_1=[1 for iU in range(self.amount_SN)]#U(-1+1)
        self.U_list_UAV_2=[1 for iU in range(self.amount_SN)]#U(-1+1)

        self.state_list_3=[self.time_slot,self.agent_pos_1[0],self.agent_pos_1[1]]
        self.state_list_4=[self.time_slot,self.agent_pos_2[0],self.agent_pos_2[1]]
        
        for i in range(self.amount_SN):
          self.state_list_3.append(self.AoI_list_UAV_1[i])
          self.state_list_4.append(self.AoI_list_UAV_2[i])

        self.duration_time_of_UAV_1 = 0
        self.duration_time_of_UAV_2 = 0

        self.total_duration_time_of_UAV_1 = 0
        self.total_duration_time_of_UAV_2 = 0

        self.state_list_3 = np.array(self.state_list_3)
        self.state_list_4 = np.array(self.state_list_4)

        self.state_list = [self.time_slot,self.agent_pos_1[0], self.agent_pos_1[1],self.agent_pos_2[0],self.agent_pos_2[1]]
        for i in range(self.amount_SN):
          self.state_list.append(self.AoI_list[i])
        self.state = np.array(self.state_list)
      
        self.check_list=[]
        for i in range(amount_SN):
          self.check_list.append(0)

        self.trans_to_BS_from_UAV_1 = -1
        self.trans_to_BS_from_UAV_2 = -1

        self.fix_pos_1 = self.agent_pos_1
        self.fix_pos_2 = self.agent_pos_2

        self.local_collection_times_1 = 0
        self.local_collection_times_2 = 0

        return self.state

      # 環境の1ステップ実行 (3)
    def step(self, action_1, action_2):
        self.amount_AoI = sum(self.AoI_list)
        self.reward = self.amount_AoI / self.amount_SN

        if 300<=self.agent_pos_1[0]<=400:
          self.state_list_3 = [self.time_slot,self.agent_pos_1[0],self.agent_pos_1[1]]
          for iS in range(amount_SN):
            self.state_list_3.append(self.AoI_list_UAV_1[iS])
          self.state_list_3 = np.array(self.state_list_3)
          action_1 = self.model.test_get_action_3(self.state_list_3)
          self.duration_time_of_UAV_1=1
          self.total_duration_time_of_UAV_1 += 1
        else:
          for iS in range(amount_SN):
            self.AoI_list_UAV_1[iS] = min(self.AoI_list[iS],self.AoI_list_UAV_1[iS])
          self.duration_time_of_UAV_1 = 0

        #遠隔制御判定_UAV_2
        if 300<=self.agent_pos_2[0]<=400:
          self.state_list_4 = [self.time_slot,self.agent_pos_2[0],self.agent_pos_2[1]]
          for iS in range(amount_SN):
            self.state_list_4.append(self.AoI_list_UAV_2[iS])
          self.state_list_4 = np.array(self.state_list_4)
          action_2 = self.model.test_get_action_4(self.state_list_4)
          self.duration_time_of_UAV_2=1
          self.total_duration_time_of_UAV_2 += 1
        else:
          for iS in range(amount_SN):
            self.AoI_list_UAV_2[iS] = min(self.AoI_list[iS],self.AoI_list_UAV_2[iS])
          self.duration_time_of_UAV_2 = 0

        #遠隔制御判定
        self.reward_1=0
        self.BS_reward_1=0

        #agent1が情報収集するか問題S(n)
        for iS in range(self.amount_SN):
          if (self.agent_pos_1[0]==self.SNposition_list[iS][0] and self.agent_pos_1[1]==self.SNposition_list[iS][1] and self.U_list[iS] >0):
            self.S_list_1[iS] = 1
            self.check_list[iS] +=1
            self.D_1_list.append([self.time_slot,iS,self.U_list[iS]])#time_slot,何番目のGNか,Uの値.
            if self.duration_time_of_UAV_1 >0 :
              if self.AoI_list_UAV_1[iS]-self.U_list[iS]>0:
                self.reward_1=self.AoI_list_UAV_1[iS]-self.U_list[iS]              
              if self.AoI_list[iS]-self.U_list[iS]>0:
                self.BS_reward_1 = self.AoI_list[iS]-self.U_list[iS]
              self.local_collection_times_1 +=1
            self.AoI_list_UAV_1[iS] = self.U_list[iS]
            self.U_list[iS] = 0
          else:
            self.S_list_1[iS] = 0
            self.AoI_list_UAV_1[iS]+=1

        self.reward_2=0
        self.BS_reward_2=0
        for iS in range(self.amount_SN):
          if (self.agent_pos_2[0]==self.SNposition_list[iS][0] and self.agent_pos_2[1]==self.SNposition_list[iS][1] and self.U_list[iS] >0):
            self.S_list_2[iS] = 1
            self.check_list[iS] +=1
            self.D_2_list.append([self.time_slot,iS,self.U_list[iS]])
            if self.duration_time_of_UAV_1 >0 :
              if self.AoI_list_UAV_2[iS]-self.U_list[iS]>0:
                self.reward_2=self.AoI_list_UAV_2[iS]-self.U_list[iS]              
              if self.AoI_list[iS]-self.U_list[iS]>0:
                self.BS_reward_2 = self.AoI_list[iS]-self.U_list[iS]
              self.local_collection_times_2 +=1
            self.AoI_list_UAV_2[iS] = self.U_list[iS]
            self.U_list[iS] = 0
          else:
            self.S_list_2[iS] = 0
            self.AoI_list_UAV_2[iS]+=1
        for iO in range(self.amount_SN):
          self.big_o_list[iO] = min((self.big_o_list[iO]+self.o_list[iO]),1)-max(self.S_list_1[iO],self.S_list_2[iO])#ここはS(n)だよ！not S(n+1)
        
        self.trans_to_BS_from_UAV_1 = -1

        self.AoI_list_BS_1 = self.AoI_list
        if self.duration_time_of_UAV_1==0:
          self.AoI_list_BS_1 = self.AoI_list
          if len(self.D_1_list) > 0:#これによってデータを保持していることがわかる！
            #どのデータを送るのか。
            #timeslot判定[タイムスロット、GNの順番、データの経過時間]
            if self.time_slot == self.D_1_list[len(self.D_1_list)-1][0]:
              ##以下のコードでBSに送信していることとなる！！#AoIのベースセンターを設定している場所！
              self.AoI_list_BS_1[self.D_1_list[len(self.D_1_list)-1][1]] = self.D_1_list[len(self.D_1_list)-1][2]
              self.trans_to_BS_from_UAV_1 = self.D_1_list[len(self.D_1_list)-1][1]
              del self.D_1_list[len(self.D_1_list)-1]
            else:#ここで同列立場のコーディング！今のAoI値との差の大きいものを送信する。

              self.D_1_list_change=[]
              for i in range(len(self.D_1_list)):#AoIのリスト(今ある保持のデータ)と対象データとの差
                self.difference_1 = self.AoI_list_BS_1[self.D_1_list[i][1]]-self.D_1_list[i][2]
                if self.difference_1 > 0:#[何番目のデータか、GN、差]
                  self.append_list =[self.D_1_list[i][0],self.D_1_list[i][1],self.D_1_list[i][2],self.difference_1]
                  self.D_1_list_change.append(self.append_list)
                  
              #listから最大値を参照する！
              #リストからの設定をしました[np.argmin(リスト、axis=0)[1]]
              #上記の要素を選択!
              self.D_1_list=self.D_1_list_change

              if len(self.D_1_list) > 0:
                self.timeslot_index = np.argmax(self.D_1_list,axis=0)[3]
                #ここで送信している！
                self.GN_1_index = self.D_1_list[self.timeslot_index][1]
                self.AoI_list_BS_1[self.GN_1_index] = self.D_1_list[self.timeslot_index][2]
                self.trans_to_BS_from_UAV_1 = self.GN_1_index
                del self.D_1_list[self.timeslot_index]
          else:
            pass
        else:
          pass

        self.trans_to_BS_from_UAV_2 = -1

        self.AoI_list_BS_2 = self.AoI_list
        if self.duration_time_of_UAV_2==0:
          self.AoI_list_BS_2 = self.AoI_list
          if len(self.D_2_list) > 0:#これによってデータを保持していることがわかる！
            #どのデータを送るのか。
            #timeslot判定[タイムスロット、GNの順番、データの経過時間]
            if self.time_slot == self.D_2_list[len(self.D_2_list)-1][0]:
              ##以下のコードでBSに送信していることとなる！！#AoIのベースセンターを設定している場所！
              self.AoI_list_BS_2[self.D_2_list[len(self.D_2_list)-1][1]] = self.D_2_list[len(self.D_2_list)-1][2]
              self.trans_to_BS_from_UAV_2 = self.D_2_list[len(self.D_2_list)-1][1]
              del self.D_2_list[len(self.D_2_list)-1]
            else:#ここで同列立場のコーディング！今のAoI値との差の大きいものを送信する。

              self.D_2_list_change=[]
              for i in range(len(self.D_2_list)):#AoIのリスト(今ある保持のデータ)と対象データとの差
                self.difference_2 = self.AoI_list_BS_2[self.D_2_list[i][1]]-self.D_2_list[i][2]
                if self.difference_2 > 0:#[何番目のデータか、GN、差]
                  self.append_list =[self.D_2_list[i][0],self.D_2_list[i][1],self.D_2_list[i][2],self.difference_2]
                  self.D_2_list_change.append(self.append_list)
                  
              #listから最大値を参照する！
              #リストからの設定をしました[np.argmin(リスト、axis=0)[1]]
              #上記の要素を選択!
              self.D_2_list=self.D_2_list_change

              if len(self.D_2_list) > 0:
                self.timeslot_index = np.argmax(self.D_2_list,axis=0)[3]
                #ここで送信している！
                self.GN_2_index = self.D_2_list[self.timeslot_index][1]
                self.AoI_list_BS_2[self.GN_2_index] = self.D_2_list[self.timeslot_index][2]
                self.trans_to_BS_from_UAV_1 = self.GN_2_index
                del self.D_2_list[self.timeslot_index]
          else:
            pass
        else:
          pass

          #AoIの更新を行う。
            #どちらも送れていない時。
            #片方のみ送れている時。
        for iA in range(self.amount_SN):
          if self.trans_to_BS_from_UAV_1 != iA and self.trans_to_BS_from_UAV_2 != iA:
            self.AoI_list[iA] += 1
          else: 
            self.AoI_list[iA] = min(self.AoI_list_BS_1[iA],self.AoI_list_BS_2[iA])

        self.time_slot += 1##ここですここ！！マスが切り替わるところです！！

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
        
        for io in range(self.amount_SN):
          self.probability_for_exponential_distribution=np.random.random()
          self.threshold_probability = self.exponential_distribution_list[self.SN_distribution_list[io]-1][self.time_slot-self.previous_time_slot_list[io]]
          if self.probability_for_exponential_distribution < self.threshold_probability:
            self.o_list[io]=1
            self.previous_time_slot_list[io]=self.time_slot
          else:
            self.o_list[io]=0

        for iU in range(self.amount_SN):
          if (self.big_o_list[iU]==0 and self.o_list[iU]==0):#O(n)=0かつo(n+1)=0
            self.U_list[iU]=0
          elif self.o_list[iU]==1:
            self.U_list[iU]=1
          else:
            self.U_list[iU]= self.U_list[iU] + 1
        for iA in range(self.amount_SN):
          self.AoI_list[iA] = np.clip(self.AoI_list[iA], 0, self.max_AoI)

        self.done = self.time_slot == 100##100回リワードを出力したことになる！
        #if self.done == True :
          #print(self.D_1_list,self.D_2_list)
          #print(self.check_list)
          #print(self.AoI_list_UAV_1)
          #print(self.AoI_list_UAV_2)
          #print(self.AoI_list_BS_1)
          #print(self.AoI_list_BS_2)
          #print(self.AoI_list)
          #print(self.duration_time_of_UAV_1)
          #print(self.duration_time_of_UAV_2)

        #遠隔制御判定_UAV_1
        self.state_AoI_list = self.AoI_list
        if 300<=self.agent_pos_1[0]<=400:
          self.duration_time_of_UAV_1=1
        else:
          self.duration_time_of_UAV_1=0
        if self.duration_time_of_UAV_1==0:
          self.fix_pos_1 = [self.agent_pos_1[0],self.agent_pos_1[1]]
          for iS in range(self.amount_SN):
            self.state_AoI_list[iS]=min(self.AoI_list_UAV_1[iS],self.state_AoI_list[iS])
        if 300<=self.agent_pos_2[0]<=400:
          self.duration_time_of_UAV_2=1
        else:
          self.duration_time_of_UAV_2=0
        if self.duration_time_of_UAV_2==0:
          self.fix_pos_2 = [self.agent_pos_2[0],self.agent_pos_2[1]]
          for iS in range(self.amount_SN):
            self.state_AoI_list[iS]=min(self.AoI_list_UAV_2[iS],self.state_AoI_list[iS])

        #print(self.agent_pos_1[0],self.agent_pos_1[1],self.agent_pos_2[0],self.agent_pos_2[1],self.duration_time_of_UAV_1,self.duration_time_of_UAV_2)
        #print(self.AoI_list_UAV_1,self.AoI_list_UAV_2)

        #UAV①内でのAoI値のステップ更新
        if len(self.D_1_list)>0:
          for i in range(len(self.D_1_list)):
            self.D_1_list[i][2]+=1
        #UAV②内でのAoI値のステップ更新
        if len(self.D_2_list)>0:
          for i in range(len(self.D_2_list)):
            self.D_2_list[i][2]+=1
        self.state_list=[self.time_slot,self.fix_pos_1[0],self.fix_pos_1[1],self.fix_pos_2[0],self.fix_pos_2[1]]

        for i in range(self.amount_SN):
          self.state_list.append(self.state_AoI_list[i])
        self.next_state = np.array(self.state_list)
        self.data_list = [[self.total_duration_time_of_UAV_1,self.local_collection_times_1,self.reward_1,self.BS_reward_1,self.D_1_list],[self.total_duration_time_of_UAV_2,self.local_collection_times_2,self.reward_2,self.BS_reward_2,self.D_2_list],self.check_list]
        #print()
        #print(self.next_state)
        #print(self.state_AoI_list)
        #print(self.AoI_list)
        #print(self.agent_pos_1,self.agent_pos_2)

        return self.next_state, self.reward, self.done, self.AoI_list,self.U_list, self.data_list,{}

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


class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim  = state_dim
        self.action_dim = aciton_dim

        self.model_1 = self.create_model_1()
        self.model_2 = self.create_model_1()
        self.model_3 = self.create_model_2()
        self.model_4 = self.create_model_2()

    def create_model_1(self):
        model = tf.keras.Sequential([
            InputLayer((self.state_dim,)),
            Dense(200, activation='relu'),
            Dense(200, activation='relu'),
            Dense(200, activation='relu'),
            Dense(self.action_dim)
        ])
        model.compile(loss='mse',metrics='mae', optimizer=Adam(args.lr))
        return model

    def create_model_2(self):
        model = tf.keras.Sequential([
            InputLayer((self.state_dim-2,)),
            Dense(200, activation='relu'),
            Dense(200, activation='relu'),
            Dense(200, activation='relu'),
            Dense(self.action_dim)
        ])
        model.compile(loss='mse',metrics='mae', optimizer=Adam(args.lr))
        return model
    
    def test_get_action_1(self, state):
        state = np.reshape(state, [1, self.state_dim])
        res_actions_list = res_actions(state[0][1],state[0][2])
        q_value = self.model_1.predict_on_batch(state)[0]#まじでここpredictじゃない方がええ気がしてますぅ。        
        res_outputs = res_output_layers(q_value,res_actions_list)
        action = choose_res_action(res_outputs)
        return action#argminへの変更も忘れずに！！]

    def test_get_action_2(self, state):
        state = np.reshape(state, [1, self.state_dim])
        res_actions_list = res_actions(state[0][3],state[0][4])
        q_value = self.model_2.predict_on_batch(state)[0]#まじでここpredictじゃない方がええ気がしてますぅ。        
        res_outputs = res_output_layers(q_value,res_actions_list)
        action = choose_res_action(res_outputs)
        return action#argminへの変更も忘れずに！！]

    def test_get_action_3(self, state):
        state = np.reshape(state, [1, self.state_dim-2])
        res_actions_list = res_actions(state[0][1],state[0][2])
        q_value = self.model_3.predict_on_batch(state)[0]#まじでここpredictじゃない方がええ気がしてますぅ。        
        res_outputs = res_output_layers(q_value,res_actions_list)
        action = choose_res_action(res_outputs)
        return action#argminへの変更も忘れずに！！]

    def test_get_action_4(self, state):
        state = np.reshape(state, [1, self.state_dim-2])
        res_actions_list = res_actions(state[0][1],state[0][2])
        q_value = self.model_4.predict_on_batch(state)[0]#まじでここpredictじゃない方がええ気がしてますぅ。        
        res_outputs = res_output_layers(q_value,res_actions_list)
        action = choose_res_action(res_outputs)
        return action#argminへの変更も忘れずに！！]       
      
class Agent:
    result_dir=result_dir
    def __init__(self, env):
        self.result_dir_1 = result_dir_1
        self.result_dir = result_dir
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.file_name_1 = file_name_1

        self.file_name = file_name


    def test_1(self, max_episodes=10):
        self.model.model_1.load_weights(os.path.join(self.result_dir_1, str(self.file_name_1)+"_model_1.h5"))
        self.model.model_2.load_weights(os.path.join(self.result_dir_1, str(self.file_name_1)+"_model_2.h5"))
        total_test_reward_list=[]
        total_average_AoI_list=[]
        total_time_UAV_1 = []
        total_time_UAV_2 = []

        total_average_reward_1_duration_time=[]
        total_average_BS_reward_1_duration_time=[]
        total_average_reward_1_collection_times=[]
        total_average_BS_reward_1_collection_times=[]
        total_proportion_collection_times_1_to_duration_time = []

        total_average_reward_2_duration_time=[]
        total_average_BS_reward_2_duration_time=[]
        total_average_reward_2_collection_times=[]
        total_average_BS_reward_2_collection_times=[]
        total_proportion_collection_times_2_to_duration_time = []

        total_episode_data_list=[]
        for ep in range(max_episodes):
            total_test_reward = 0
            total_average_AoI = 0
            sum_reward_1 = 0
            sum_reward_2 = 0
            sum_BS_reward_1 = 0
            sum_BS_reward_2 = 0
            done = False
            state = self.env.reset()
            step_count=0
            AoI_list=[int(state[1+2*2+i]) for i in range(amount_SN)]
            while not done:
                AoI_list = [int(AoI_list[i]) for i in range(amount_SN)]
                average_AoI = sum(AoI_list) / amount_SN
                action_1 = self.model.test_get_action_1(state)
                action_2 = self.model.test_get_action_2(state)
                next_state, reward, done, AoI_list, U_list, data,_ = self.env.step(action_1,action_2)
                state = next_state
                total_test_reward += reward
                sum_reward_1 += data[0][2]
                sum_BS_reward_1 += data[0][3]
                #print(data[0][3])
                sum_reward_2 += data[1][2]
                sum_BS_reward_2 += data[1][3]
                #print(data[1][3])
                total_average_AoI += average_AoI
                step_count+=1

            episode_data_list=[total_test_reward/step_count,total_average_AoI/step_count,[data[0][0],data[0][1],sum_reward_1,sum_BS_reward_1,data[0][4]],[data[1][0],data[1][1],sum_reward_2,sum_BS_reward_2,data[1][4]],data[2]]

            total_episode_data_list.append(episode_data_list)
            total_test_reward_list.append(total_test_reward/step_count)
            total_average_AoI_list.append(total_average_AoI/step_count)
            total_time_UAV_1.append(data[0][0])
            total_time_UAV_2.append(data[1][0])
            
            #duration_time
            average_reward_1_duration_time = None
            average_reward_1_duration_time = None
            proportion_collection_times_1_to_duration_time = None
            if data[0][0] > 0:
              average_reward_1_duration_time=sum_reward_1/data[0][0]
              average_BS_reward_1_duration_time=sum_BS_reward_1/data[0][0]
              total_average_reward_1_duration_time.append(average_reward_1_duration_time)
              total_average_BS_reward_1_duration_time.append(average_BS_reward_1_duration_time)            
              proportion_collection_times_1_to_duration_time=data[0][1]/data[0][0]
              total_proportion_collection_times_1_to_duration_time.append(proportion_collection_times_1_to_duration_time)

            average_reward_1_collection_times = None
            average_reward_1_collection_times = None
            if data[0][1] > 0:
              average_reward_1_collection_times=sum_reward_1/data[0][1]
              average_BS_reward_1_collection_times=sum_BS_reward_1/data[0][1]
              total_average_reward_1_collection_times.append(average_reward_1_collection_times)
              total_average_BS_reward_1_collection_times.append(average_BS_reward_1_collection_times)

            average_reward_2_duration_time = None
            average_reward_2_duration_time = None
            proportion_collection_times_2_to_duration_time = None
            if data[1][0] > 0:
              average_reward_2_duration_time=sum_reward_2/data[1][0]
              average_BS_reward_2_duration_time=sum_BS_reward_2/data[1][0]
              total_average_reward_2_duration_time.append(average_reward_2_duration_time)
              total_average_BS_reward_2_duration_time.append(average_BS_reward_2_duration_time)            
              proportion_collection_times_2_to_duration_time=data[1][1]/data[1][0]
              total_proportion_collection_times_2_to_duration_time.append(proportion_collection_times_2_to_duration_time)

            average_reward_2_collection_times = None
            average_reward_2_collection_times = None
            if data[1][1] > 0:
              average_reward_2_collection_times=sum_reward_2/data[1][1]
              average_BS_reward_2_collection_times=sum_BS_reward_2/data[1][1]
              total_average_reward_2_collection_times.append(average_reward_2_collection_times)
              total_average_BS_reward_2_collection_times.append(average_BS_reward_2_collection_times)


            ######AoI_list=[int(state[1+2*2+i]) for i in range(amount_SN)]
            
            #時間①、時間100-①、リワード①、時間②、時間100-②、リワード②、AoIの値
            #U_list=[int(state[6+amount_SN+i]) for i in range(amount_SN)]
            #U_list=[int(state[6+amount_SN+iU]) for iU in range(amount_SN)]
            #print('UAV_1:{}①:{}②:{}③:{}④:{}⑤:{}'.format(data[0][0],average_reward_1_duration_time,average_BS_reward_1_duration_time,average_reward_1_collection_times,average_BS_reward_1_collection_times,proportion_collection_times_1_to_duration_time))
            #print('UAV_2:{}①:{}②:{}③:{}④:{}⑤:{}'.format(data[1][0],average_reward_2_duration_time,average_BS_reward_2_duration_time,average_reward_2_collection_times,average_BS_reward_2_collection_times,proportion_collection_times_2_to_duration_time))
            #print('ep{} AoI/U={} / {} state_pos={} total_average_AoI={}'.format(ep, AoI_list,  U_list,[state[1],state[2],state[3],state[4]], total_average_AoI/step_count))
            #print()
            #print()


            #print('↑total_time_UAV_1',total_UAV_time_1,sum_reward_1,sum_reward_1/total_UAV_time_1,'↑total_time_UAV_2',total_UAV_time_2,sum_reward_2,sum_reward_2/total_UAV_time_2)
        #UAV_1_reward=0
        #UAV_2_reward=0
        #UAV_1_time = 0
        #UAV_2_time = 0
        #for i in range(len(data_list)):
            #total_average_reward_1_duration_time=[]
            #total_average_BS_reward_1_duration_time=[]
            #total_average_reward_1_collection_times=[]
            #total_average_BS_reward_1_collection_times=[]
            #total_proportion_collection_times_1_to_duration_time = []
        #print('UAV_1:{}①:{}②:{}③:{}④:{}⑤:{}'.format(statistics.mean(total_time_UAV_1),statistics.mean(total_average_reward_1_duration_time),statistics.mean(total_average_BS_reward_1_duration_time),statistics.mean(total_average_reward_1_collection_times),statistics.mean(total_average_BS_reward_1_collection_times),statistics.mean(total_proportion_collection_times_1_to_duration_time)))
        #print('UAV_2:{}①:{}②:{}③:{}④:{}⑤:{}'.format(statistics.mean(total_time_UAV_2),statistics.mean(total_average_reward_2_duration_time),statistics.mean(total_average_BS_reward_2_duration_time),statistics.mean(total_average_reward_2_collection_times),statistics.mean(total_average_BS_reward_2_collection_times),statistics.mean(total_proportion_collection_times_2_to_duration_time)))
        #print()
        con_data =[[statistics.mean(total_test_reward_list),min(total_test_reward_list),max(total_test_reward_list)],[statistics.mean(total_time_UAV_1),statistics.mean(total_average_reward_1_duration_time),statistics.mean(total_average_BS_reward_1_duration_time),statistics.mean(total_average_reward_1_collection_times),statistics.mean(total_average_BS_reward_1_collection_times),statistics.mean(total_proportion_collection_times_1_to_duration_time)],[statistics.mean(total_time_UAV_2),statistics.mean(total_average_reward_2_duration_time),statistics.mean(total_average_BS_reward_2_duration_time),statistics.mean(total_average_reward_2_collection_times),statistics.mean(total_average_BS_reward_2_collection_times),statistics.mean(total_proportion_collection_times_2_to_duration_time)]]
        print(con_data)

        #print('average_episode_test_reward:'+str(sum(total_test_reward_list)/max_episodes))
        #print('average_episode_sum_AoI:'+str(sum(total_average_AoI_list)/max_episodes))
        #print('min_average_AoI:',min(total_test_reward_list))
        #print('max_average_AoI:',max(total_test_reward_list))

        #statistics.mean()

        f = open(os.path.join(self.result_dir,'data_user_test.txt'),'a')
        #f.write(str(AoI_list)+'\n')
        f.write('total_episode_data_list')
        f.write(str(total_episode_data_list)+'\n')
        f.write('total_episode_reward_list\n')
        f.write(str(total_test_reward_list)+'\n')
        f.write('total_average_AoI_list\n')
        f.write(str(total_average_AoI_list)+'\n')
        f.write('average_episodes_test_reward:'+str(statistics.mean(total_test_reward_list))+'\n')
        f.write('average_episodes_average_AoI:'+str(statistics.mean(total_test_reward_list))+'\n')
        #f.write('data_list\n')
        #f.write('#時間①、時間100-①、リワード①、時間②、時間100-②、リワード②、AoIの値\n')
        #f.write(str(data_list))
        f.write('UAV_1:\n')
        f.write(str([statistics.mean(total_time_UAV_1),statistics.mean(total_average_reward_1_duration_time),statistics.mean(total_average_BS_reward_1_duration_time),statistics.mean(total_average_reward_1_collection_times),statistics.mean(total_average_BS_reward_1_collection_times),statistics.mean(total_proportion_collection_times_1_to_duration_time)])+'\n')
        f.write('UAV_2:\n')
        f.write(str([statistics.mean(total_time_UAV_2),statistics.mean(total_average_reward_2_duration_time),statistics.mean(total_average_BS_reward_2_duration_time),statistics.mean(total_average_reward_2_collection_times),statistics.mean(total_average_BS_reward_2_collection_times),statistics.mean(total_proportion_collection_times_2_to_duration_time)])+'\n')
        f.write('con_data:\n')
        f.write(str(con_data))
        f.close()
        
file_name_1 = '5_0_e_4_timeslot'
file_name_2 = '1_0_e_3_timeslot'
file_name = 'another_1000_'+str(int(100*args.threshold))+'_'+str(args.duration_threshold)+'_first_and_first'
result_dir_1 = '/content/drive/MyDrive/modi_env/DQN_for_MARL/epsilon/SN'+str(args.amount_SN)+'/'+str(file_name_1)
result_dir_2 = '/content/drive/MyDrive/modi_env/no_DQN_for_MARL/epsilon/SN'+str(args.amount_SN)+'/'+str(file_name_2)
result_dir = '/content/drive/MyDrive/modi_env/total_system/SN'+str(args.amount_SN)+'/DQN_for_MARL_and_No_DQN_for_MARL/'+str(file_name)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
def main_1_1():
    env = AoI()#ここを私のに変えましょうぜい✊✊✊
    agent = Agent(env)
    agent.test_1(max_episodes=1000)
if __name__ == "__main__":
   main_1_1()
file_name_2 = '1_0_e_3_timeslot_again'
file_name = 'another_1000_'+str(int(100*args.threshold))+'_'+str(args.duration_threshold)+'_first_and_second'
result_dir_2 = '/content/drive/MyDrive/modi_env/no_DQN_for_MARL/epsilon/SN'+str(args.amount_SN)+'/'+str(file_name_2)
result_dir = '/content/drive/MyDrive/modi_env/total_system/SN'+str(args.amount_SN)+'/DQN_for_MARL_and_No_DQN_for_MARL/'+str(file_name)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
def main_1_2():
    env = AoI()#ここを私のに変えましょうぜい✊✊✊
    agent = Agent(env)
    agent.test_1(max_episodes=1000)
if __name__ == "__main__":
   main_1_2()

file_name_2 = '1_0_e_3_timeslot_third'
file_name = 'another_1000_'+str(int(100*args.threshold))+'_'+str(args.duration_threshold)+'_first_and_third'
result_dir_2 = '/content/drive/MyDrive/modi_env/no_DQN_for_MARL/epsilon/SN'+str(args.amount_SN)+'/'+str(file_name_2)
result_dir = '/content/drive/MyDrive/modi_env/total_system/SN'+str(args.amount_SN)+'/DQN_for_MARL_and_No_DQN_for_MARL/'+str(file_name)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
def main_1_3():
    env = AoI()#ここを私のに変えましょうぜい✊✊✊
    agent = Agent(env)
    agent.test_1(max_episodes=1000)
if __name__ == "__main__":
   main_1_3()



file_name_1 = '5_0_e_4_timeslot_again'
file_name_2 = '1_0_e_3_timeslot'
file_name = 'another_1000_'+str(int(100*args.threshold))+'_'+str(args.duration_threshold)+'_second_and_first'
result_dir_1 = '/content/drive/MyDrive/modi_env/DQN_for_MARL/epsilon/SN'+str(args.amount_SN)+'/'+str(file_name_1)
result_dir_2 = '/content/drive/MyDrive/modi_env/no_DQN_for_MARL/epsilon/SN'+str(args.amount_SN)+'/'+str(file_name_2)
result_dir = '/content/drive/MyDrive/modi_env/total_system/SN'+str(args.amount_SN)+'/DQN_for_MARL_and_No_DQN_for_MARL/'+str(file_name)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
def main_2_1():
    env = AoI()#ここを私のに変えましょうぜい✊✊✊
    agent = Agent(env)
    agent.test_1(max_episodes=1000)
if __name__ == "__main__":
   main_2_1()

file_name_2 = '1_0_e_3_timeslot_again'
file_name = 'another_1000_'+str(int(100*args.threshold))+'_'+str(args.duration_threshold)+'_second_and_second'
result_dir_2 = '/content/drive/MyDrive/modi_env/no_DQN_for_MARL/epsilon/SN'+str(args.amount_SN)+'/'+str(file_name_2)
result_dir = '/content/drive/MyDrive/modi_env/total_system/SN'+str(args.amount_SN)+'/DQN_for_MARL_and_No_DQN_for_MARL/'+str(file_name)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
def main_2_2():
    env = AoI()#ここを私のに変えましょうぜい✊✊✊
    agent = Agent(env)
    agent.test_1(max_episodes=1000)
if __name__ == "__main__":
   main_2_2()

file_name_2 = '1_0_e_3_timeslot_third'
file_name = 'another_1000_'+str(int(100*args.threshold))+'_'+str(args.duration_threshold)+'_second_and_third'
result_dir_2 = '/content/drive/MyDrive/modi_env/no_DQN_for_MARL/epsilon/SN'+str(args.amount_SN)+'/'+str(file_name_2)
result_dir = '/content/drive/MyDrive/modi_env/total_system/SN'+str(args.amount_SN)+'/DQN_for_MARL_and_No_DQN_for_MARL/'+str(file_name)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
def main_2_3():
    env = AoI()#ここを私のに変えましょうぜい✊✊✊
    agent = Agent(env)
    agent.test_1(max_episodes=1000)
if __name__ == "__main__":
   main_2_3()



file_name_1 = '5_0_e_4_timeslot_third'
file_name_2 = '1_0_e_3_timeslot'
file_name = 'another_1000_'+str(int(100*args.threshold))+'_'+str(args.duration_threshold)+'_third_and_first'
result_dir_1 = '/content/drive/MyDrive/modi_env/DQN_for_MARL/epsilon/SN'+str(args.amount_SN)+'/'+str(file_name_1)
result_dir_2 = '/content/drive/MyDrive/modi_env/no_DQN_for_MARL/epsilon/SN'+str(args.amount_SN)+'/'+str(file_name_2)
result_dir = '/content/drive/MyDrive/modi_env/total_system/SN'+str(args.amount_SN)+'/DQN_for_MARL_and_No_DQN_for_MARL/'+str(file_name)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
def main_3_1():
    env = AoI()#ここを私のに変えましょうぜい✊✊✊
    agent = Agent(env)
    agent.test_1(max_episodes=1000)
if __name__ == "__main__":
   main_3_1()

file_name_2 = '1_0_e_3_timeslot_again'
file_name = 'another_1000_'+str(int(100*args.threshold))+'_'+str(args.duration_threshold)+'_third_and_second'
result_dir_2 = '/content/drive/MyDrive/modi_env/no_DQN_for_MARL/epsilon/SN'+str(args.amount_SN)+'/'+str(file_name_2)
result_dir = '/content/drive/MyDrive/modi_env/total_system/SN'+str(args.amount_SN)+'/DQN_for_MARL_and_No_DQN_for_MARL/'+str(file_name)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
def main_3_2():
    env = AoI()#ここを私のに変えましょうぜい✊✊✊
    agent = Agent(env)
    agent.test_1(max_episodes=1000)
if __name__ == "__main__":
   main_3_2()

file_name_2 = '1_0_e_3_timeslot_third'
file_name = 'another_1000_'+str(int(100*args.threshold))+'_'+str(args.duration_threshold)+'_third_and_third'
result_dir_2 = '/content/drive/MyDrive/modi_env/no_DQN_for_MARL/epsilon/SN'+str(args.amount_SN)+'/'+str(file_name_2)
result_dir = '/content/drive/MyDrive/modi_env/total_system/SN'+str(args.amount_SN)+'/DQN_for_MARL_and_No_DQN_for_MARL/'+str(file_name)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
def main_3_3():
    env = AoI()#ここを私のに変えましょうぜい✊✊✊
    agent = Agent(env)
    agent.test_1(max_episodes=1000)
if __name__ == "__main__":
   main_3_3()
