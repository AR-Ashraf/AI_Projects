import numpy as np
import random
R = np.matrix([
    [0,0,0,0,1,0],
    [0,0,0,1,0,1],
    [0,0,100,1,0,0],
    [0,1,1,0,1,0],
    [1,0,0,1,0,0],
    [0,1,0,0,0,0]
])

Q = np.matrix(np.zeros([6,6]))

def possible_actions(state):
    current_state_row = R[state,]
    #print("current state row: " , current_state_row)
    possible_act = np.where(current_state_row>0)[1]
    return possible_act


def ActionChoice(available_action_range):
    next_action = int(np.random.choice(available_action_range,1))
    return next_action

def reward(current_state,action,gamma):
    Max_state = np.where(Q[action,] == np.max(Q[action,]))[1]

    if Max_state.shape[0] > 1:
        Max_state = int(np.random.choice(Max_state,size=1))
    else:
        Max_state = int(Max_state)
    MaxValue = Q[action,Max_state]
    Q[current_state,action] = R[current_state,action] + gamma * MaxValue

for i in range(50000):
    current_state = np.random.randint(0,int(Q.shape[0]))
    PossibleActions = possible_actions(current_state)
    action = ActionChoice(PossibleActions)
    reward(current_state,action,.8)

print("Q: ")
print(Q)
print("Nomed Q: ")
print(Q/np.max(Q)*100)