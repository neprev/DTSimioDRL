from DQNagent import DQNagent, GetlastID
import pandas as pd
import pyodbc
import csv
import dill
import h5py



#op
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-TJNK3RM;DATABASE=DRL;Trusted_Connection=yes;')
cursor = conn.cursor()

#Importing statr data from the database
StateDf = pd.read_sql("SELECT * From StateDRL", conn)

#done file for knowing the ID
with open('E:\pfe\Done.csv', 'r') as csv_file:
    rowlist = []
    csvreader = csv.reader(csv_file)
    for row in csvreader:
        rowlist.append(row)
    ID=int(rowlist[-1][0])
    print(ID)

#Fixing the state size and the action size
state_size = 5
action_size = 2

#defining the DQN agent and the parameters
agent = DQNagent(state_size, action_size)
done = False
batch_size = 32

#need to clean data before this:
#putting the state table in Numpy form
stateNP = StateDf.to_numpy()

#extracting the current state
state = stateNP [ID+1][1:5]
print (state)

#taking the action
action = agent.act(state)
print(action)

#saving the state and the action as an attribute to the agent
agent.last_iteration.append([ID,state,action])
print(agent.last_iteration)

#saving the object as a pickle object
agent_outfile = open("DQNagent.joblib", "wb")
dill.dump(agent, agent_outfile)
agent_outfile.close()

#saving action to csv file
cursor.execute("INSERT INTO Go1 (Action) VALUES ({})".format(ID, action))

