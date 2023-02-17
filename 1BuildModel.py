from DQNagent import DQNagent
import pyodbc
from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam
from keras.models import save_model
import dill

#op
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-TJNK3RM;DATABASE=DRL;Trusted_Connection=yes;')
cursor = conn.cursor()

#Fixing the state size and the action size
state_size = 4
action_size = 2

#defining the DQN agent and the parameters
agent = DQNagent(state_size, action_size)

def _build_model(agent):
    model = Sequential()
    model.add(Dense(24, input_dim=agent.state_size, activation=relu))
    model.add(Dense(24, activation=relu))
    model.add(Dense(agent.action_size, activation=linear))
    model.compile(loss="mse", optimizer=Adam(learning_rate=agent.learning_rate))
    return model


model = _build_model(agent)
print(agent.memory)

model.summary()

save_model(model, "model6.h5")
agent_outfile = open(r"C:\Users\DELL\PycharmProjects\pythonProject\DQNagent6.joblib", "wb")
dill.dump(agent, agent_outfile)
agent_outfile.close()

