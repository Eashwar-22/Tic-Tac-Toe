import numpy as np
import pandas as pd
import pathlib
import os
import glob
from time import sleep,time
from datetime import datetime
from IPython.core import display as ICD

import warnings
warnings.filterwarnings("ignore")

class tictactoe:
    # INITIALIZE CLASS OBJECT
    def __init__(self,
                 model=None
                 ,  #trained model
                 path=str(pathlib.Path().resolve())+'/Database/',
                ):
        self.model=model
        self.path=path
        self.matrix="----------------\n|  1 | 2  | 3  |\n----------------\n| 4  | 5  | 6  |\n----------------\n| 7  | 8  | 9  |\n----------------\n"
        self.turn_map={'X':1,'O':-1}
        self.win_combinations=[[1,2,3],[4,5,6],[7,8,9],
                               [1,4,7],[2,5,8],[3,6,9],
                               [1,5,9],[3,5,7]]
        self.cell_index={'1':20,'2':24,'3':29,
                         '4':53,'5':58,'6':63,
                         '7':87,'8':92,'9':97}
        self.reset_data()
        self.reset_game()
        
    # UPDATE VERBOSE
    def replace_cell_display(self,index):
        i=str(index)
        self.play_matrix=self.play_matrix[:self.cell_index[i]] + self.turn + self.play_matrix[self.cell_index[i]+1:]
    
    # UPDATE ACTIVE CELLS 
    def replace_cell_array(self,index):
        self.arr[index-1]=self.turn_map[self.turn]
    
    # MONITOR WIN CRITERIA
    def check_win_criteria(self):
        for i in self.win_combinations:
            if all(self.arr[index-1] == self.turn_map[self.turn] for index in i):
                return 1
        return 0
    
    # NEW GAME
    def reset_game(self):
        self.play_matrix = "----------------\n|    |    |    |\n----------------\n|    |    |    |\n----------------\n|    |    |    |\n----------------\n"
        self.entry_count=0
        self.turn='X'
        self.arr=np.array([0,0,0,0,0,0,0,0,0])
        self.switch_turn=0
        self.break_=False
        
    # NEW SET OF DATA    
    def reset_data(self):
        self.columns=['player','win']+[str(i)+"_played" for i in range(1,10)]+\
                     [str(i)+"_threat" for i in range(1,10)]+\
                     [str(i)+"_win" for i in range(1,10)]+['next_move']
        self.data=pd.DataFrame(columns=self.columns)
        
        print("...Reset data...")
    
    # CHECK FOR POTENTIAL LOSS / WIN (UPDATE THE 18 FLAGS -> 9+9)
    def check_status(self):
        map_check = {'X':[2,-2],
                     'O':[-2,2]}
        for c in self.win_combinations:
            l=[self.arr[i-1] for i in c]
            if sum(l)==map_check[self.turn][0]:
                self.data.iloc[self.data_index,19+c[l.index(0)]]=1 # cell for potential win
            if sum(l)==map_check[self.turn][1]:
                self.data.iloc[self.data_index,10+c[l.index(0)]]=1 # cell for potential loss
    
    # NEW GAME? (Y/N)
    def new_game_request(self):
        n = input("New game? (y/n): ")
        if n=='y':
            print("--------------------------------------------------")
            print("NEW GAME : ")
            self.reset_game()
            self.data_index+=1
        else:
            print("Thank you for playing......")
            sleep(3)
            self.break_=True
        
    # STORE GAME SESSION DATA
    def save_data(self):
        self.data.to_csv(self.path+f'Game_{str(datetime.now())}.csv')
        
    # PLAY GAME
    def play(self):
        while(1):
            self.vs_computer=input("Enter mode (c ---> vs computer, h ---> vs human ) : ")
            if self.vs_computer in ['c','h']:
                break
        
        comp_retry=0
        self.reset_game()
        self.reset_data()
        self.data_index=0
        while(1):
            print("--------------------------------------------------")
            self.data.loc[self.data.shape[0]] = [0]*len(self.data.columns)
            self.switch_turn=1
            print(self.arr)
            print(self.matrix)
            print(self.play_matrix)
            if self.entry_count==9:
                print("Draw...")
                self.new_game_request()
                if self.break_:
                    self.data['next_move']=self.data['next_move'].shift(-1)
                    if self.vs_computer=='h':
                        self.save_data()
                    break
                self.data.loc[self.data.shape[0]] = [0]*len(self.data.columns)
                self.switch_turn=1
#                 print(self.arr)
                print(self.matrix)
                print(self.play_matrix)
            if (self.turn=='O')&(self.vs_computer=='c'):
                if comp_retry==1:
                    ind = np.argmax(pred)
                    inp= np.argmax(pred)+1
                    comp_retry=0
                else:
                    model_input = np.array(self.data.iloc[-2,2:-1],dtype='float64')
                    pred=list(self.model.predict(model_input.reshape(1,-1))[0])
                    ind = np.argmax(pred)
                    inp= np.argmax(pred)+1
                
            else:
                inp=int(input(f"Enter cell index to place {self.turn} : "))
            c = str(inp)
            
            # enter 0 to stop the game
            if inp==0:
                self.new_game_request()
                if self.break_:
                    self.data['next_move']=self.data['next_move'].shift(-1)
                    if self.vs_computer=='h':
                        self.save_data()
                    break
                
            # check for invalid entries
            if inp>9:
                self.switch_turn=0
                print("try a valid cell")
            elif self.play_matrix[self.cell_index[c]] in ['X','O']:
                if (self.turn=='O')&(self.vs_computer=='c'):
                    comp_retry=1
                    pred[ind] = -99
                    ind = np.argmax(pred)
                    inp = np.argmax(pred)+1
                self.switch_turn=0
                print("try another cell")

            # operations after a valid entry
            else:
                
                self.entry_count+=1
                self.replace_cell_array(inp)
                self.replace_cell_display(c)
                self.data.iloc[self.data_index,-1]   = inp
                self.data.iloc[self.data_index,0]    = self.turn
                self.data.iloc[self.data_index,2:11] = self.arr
                self.check_status()

                if self.check_win_criteria()==1:
                    self.data.iloc[self.data_index,1]=1
                    print(self.play_matrix)
                    print(f"Player {self.turn} wins the game.")
                    self.new_game_request()
                    if self.break_:
                        self.data['next_move']=self.data['next_move'].shift(-1)
                        if self.vs_computer=='h':
                            self.save_data()
                        break


            # operations before next turn
            if self.switch_turn==1:
                self.data_index+=1
                if self.turn=='X':
                    self.turn='O'
                else:
                    self.turn='X'
                    
    # CONCAT DATABASE FILES
    def combine_existing_data(self):
        csv_files = glob.glob(os.path.join(self.path, "*.csv"))
        comb_df = pd.DataFrame(columns=self.data.columns)
        for i in csv_files:
            temp = pd.read_csv(i)
            comb_df = pd.concat([comb_df,temp],ignore_index=True)
        return comb_df
        
        
        
        
        
        