import dill
from keras.models import load_model
import random
agent_infile = open(r"C:\Users\DELL\PycharmProjects\pythonProject\DQNagent.joblib","rb")
agent = dill.load(agent_infile)
print("last iteration",agent.last_iteration)

print(agent.IDres)

f = open('experience.txt','a')
model = load_model("model.h5")

model.summary()

print("memory : \n" , agent.memory[1000])

# sum_reward = 0
# for item in agent.memory:
#     sum_reward += item[3]
#
# average_reward = sum_reward / len(agent.memory)

model.summary()


#print("episode: {}/{}, score: {}, e: {}"
#   .format(agent.episode, agent.max_episodes, average_reward, agent.epsilon))
somme= 0
batch_size=32
minibatch = random.sample(agent.memory, batch_size)
for state, action, next_state, reward in agent.memory:
    somme+=reward
print(somme/len(agent.memory))

