import dill
from DQNagent import replay
from keras.models import load_model, save_model


batch_size = 40

agent_infile = open(r"C:\Users\DELL\PycharmProjects\pythonProject\DQNagent6.joblib","rb")
agent = dill.load(agent_infile)

agent.episode += 1


# print(agent.memory)
# print(agent.last_iteration)

model = load_model(r'C:\Users\DELL\PycharmProjects\pythonProject\model6.h5')

# [print(i.shape, i.dtype) for i in model.inputs]
# [print(o.shape, o.dtype) for o in model.outputs]
# [print(l.name, l.input_shape, l.dtype) for l in model.layers]

if agent.episode < agent.max_episodes:
    if len(agent.memory) > batch_size:
        replay(agent, batch_size, model)


save_model(model, r'C:\Users\DELL\PycharmProjects\pythonProject\model6.h5')



sum_reward = 0
for item in agent.last_iteration:
    if len(item) > 3:
        sum_reward += item[4]

average_reward = sum_reward / len(agent.last_iteration)

f = open(r'C:\Users\DELL\PycharmProjects\pythonProject\experience6.txt','a')
f.write("episode: {}/{}, score: {:}, e: {:} \n"
      .format(agent.episode, agent.max_episodes, average_reward, agent.epsilon))
f.close()

agent.last_iteration = []
agent.IDres = 0
agent.ID = 0

agent_outfile = open(r"C:\Users\DELL\PycharmProjects\pythonProject\DQNagent6.joblib", "wb")
dill.dump(agent, agent_outfile)
agent_outfile.close()
