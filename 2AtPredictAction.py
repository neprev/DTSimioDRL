from DQNagent import act, clean_state
import pandas as pd
import pyodbc
import dill
from keras.models import load_model, save_model
import numpy as np

#Connect
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-TJNK3RM;DATABASE=DRL;Trusted_Connection=yes;')
cursor = conn.cursor()

#Importing state data from the database
StateDf = pd.read_sql("SELECT * From StateDRL", conn)


agent_infile = open(r"C:\Users\DELL\PycharmProjects\pythonProject\DQNagent6.joblib","rb")
agent = dill.load(agent_infile)
print(agent.last_iteration)

#putting the state table in Numpy form
stateNP = StateDf.to_numpy()
ID = agent.ID
#extracting the current state
state = clean_state(stateNP [ID-1][1:5])

#reshaping the state to the input shape of the model
state = np.reshape(state, [1, agent.state_size])

#loading the model
model = load_model(r'C:\Users\DELL\PycharmProjects\pythonProject\model6.h5')

#taking the action
action = act(agent, model, state)

#saving the state and the action as an attribute to the agent
agent.last_iteration.append([ID, state, action])
agent.ID += 1

#print(agent.last_iteration)

#saving the object as a pickle object
agent_outfile = open(r"C:\Users\DELL\PycharmProjects\pythonProject\DQNagent6.joblib", "wb")
dill.dump(agent, agent_outfile)
agent_outfile.close()

#saving action to database
cursor.execute("INSERT INTO Go1 (Action) VALUES ({})".format(action))
cursor.commit()
conn.close()

save_model (model, r"C:\Users\DELL\PycharmProjects\pythonProject\model6.h5")
