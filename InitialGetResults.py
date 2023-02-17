from DQNagent import DQNagent
import pyodbc
import pandas as pd
import dill

conn = pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-TJNK3RM;DATABASE=DRL;Trusted_Connection=yes;')
cursor = conn.cursor()

#Importing data from the database
ShipTime = pd.read_sql("SELECT * From ShipTimes", conn)
StateDf = pd.read_sql("SELECT * From StateDRL", conn)
Action = pd.read_sql("SELECT * From Go1", conn)

agent_infile = open("DQNagent.joblib","rb")
agent = dill.load(agent_infile)
print(agent.last_iteration)

"""state_size = 5
action_size = 2
done = False
batch_size = 32
"""
ID = agent.last_iteration[0]
StateNP = StateDf.to_numpy()
state = StateNP[-2][1:5]
next_state = StateNP[-1][1:5]
action = Action[-1]

Time1 = next_state[-2]
Time2 = ShipTime[-1]
Reward = 1/(Time2 - Time1)

agent.last_iteration.append(next_state,Reward)
print(agent.last_iteration)
