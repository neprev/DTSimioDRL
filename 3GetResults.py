from DQNagent import clean_state
import pyodbc
import pandas as pd
import dill
import numpy as np

conn = pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-TJNK3RM;DATABASE=DRL;Trusted_Connection=yes;')
cursor = conn.cursor()

#Importing data from the database
ShipTime = pd.read_sql("SELECT * From ShipTimes", conn)
StateDf = pd.read_sql("SELECT * From StateDRL", conn)
#Action = pd.read_sql("SELECT * From Go1", conn)

agent_infile = open(r"C:\Users\DELL\PycharmProjects\pythonProject\DQNagent6.joblib","rb")
agent = dill.load(agent_infile)
print(agent.last_iteration)

"""state_size = 5
action_size = 2
done = False
"""

thisID = agent.IDres
this_iteration = agent.last_iteration[thisID]


StateNP = StateDf.to_numpy()
state = this_iteration[1]
# print('state : ',state)
next_state = StateNP[thisID][1:5]
next_state = clean_state(next_state)
# print('next state : ',next_state)
action = this_iteration[2]
# print('action : ', action)

next_state = np.reshape(next_state, [1, agent.state_size])

Time1 = StateNP[thisID][-1]
# print('Time MPS : ',Time1)
ShipTimeNP = ShipTime.to_numpy()
Time2 = ShipTimeNP[thisID][1]
# print('Time Shipped : ', Time2)
Reward = 1/(Time2 - Time1)
# print('Reward : ', Reward)


this_iteration.append(next_state)
this_iteration.append(Reward)

agent.last_iteration[thisID] = this_iteration
# print(this_iteration)
# print(agent.last_iteration)

# print('iteration data [ID, state, action, next state, reward] : \n', agent.last_iteration[thisID])

agent.remember(this_iteration)
agent.IDres += 1

#save the agent again
agent_outfile = open(r"C:\Users\DELL\PycharmProjects\pythonProject\DQNagent6.joblib", "wb")
dill.dump(agent, agent_outfile)
agent_outfile.close()
