import numpy as np
import sys

fileName = sys.argv[1]      # Taking the file name, input as command line argument.
loop = int(sys.argv[2]) if len(sys.argv)>2 else 200         # Taking number of iterations as command line argument, default value '200'.

class LP_algo:
    
    def __init__(self, fileName):
        
        with open(f"MDPs/{fileName}") as file:

            for line in file:
                keys = line.split(" ")
                
                if keys[0] == "states":
                    self.states = int(keys[1])
                
                elif keys[0] == "actions":
                    
                    self.actions = int(keys[1])
                    # Initialising arrays for storing probability and reward.
                    self.prob = np.zeros((self.states, self.actions, self.states))
                    self.r = np.zeros((self.states, self.actions, self.states))
                
                elif keys[0] == "tran":
                    
                    # Storing probability and reward values in arrays with indexes corresponding to ( initial state , action taken , final state )
                    self.prob[int(keys[1])][int(keys[2])][int(keys[3])] = float(keys[5][:-1])
                    self.r[int(keys[1])][int(keys[2])][int(keys[3])] = float(keys[4])
                
                elif keys[0] == "gamma":
                    self.gm = float(keys[1])
    
    def my_lp(self):
        
        self.vals = np.ones(self.states, dtype=float)   # Array containing value function values for all states corresponding to it's index.
        self.policy = np.ones(self.states, dtype=int)   # Array containing optimal policy for all states corresponding to it's index.
        
        for i1 in range(loop):
            for i2 in range(self.states):
                # Implementing linear programming algorithm.
                self.vals[i2] = np.max(np.array([np.sum([ self.prob[i2][i][j]*(self.r[i2][i][j] + self.gm*self.vals[j]) for j in range(self.states)]) for i in range(self.actions)]))
        
        for st in range(self.states):
        
            # Evaluating optimal policy corresponding to the calculated value function.
            self.policy[st] = np.argmax(np.sum([[self.prob[st][i][j]*(self.r[st][i][j] + self.gm*self.vals[j]) for j in range(self.states)] for i in range(self.actions)], axis=1))

    def store(self, out):
        ''' Storing the output in format :    <value function>   <optimal policy>
                                                  .......            .......
                                                  .......            .......
            in a txt file.
        '''
        with open(f"sol_{out}", 'w') as file:
            for i in range(self.states):
                file.write(f'{round(self.vals[i], 6)} {self.policy[i]}\n')

# Running the algorithm.
algo = LP_algo(fileName)
LP_algo.my_lp(algo)
LP_algo.store(algo, fileName)