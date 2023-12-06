import gym
from run_a2c import load
from run_ma2c import loadmodel
import sys
import numpy as np
sys.path.append("../highway-env")
import highway_env

env = gym.make("merge-multi-agent-v1")
s = np.random.randint(0,1000)
s = 992
env.seed = s
env.unwrapped.seed = s
runs = 20

# model_dir = "results/at_cl/Nov-03_12_22_29/"
# model_dir = "MAA2C_5/Oct-02_12_53_20"
model_dir = "results/adv_speed_train/Nov-27_15_55_46/"
cav_model = loadmodel(model_dir, global_step=100, load_adv=False)

# adv_dir = "results/Oct-05_09_24_08"
# model = loadmodel(model_dir)
# adv_model = load(adv_dir)  
# print(model.env.observation_space)
# for i in range(10):
#     X = np.random.rand(5, 25)
#     M = np.random.rand(5, 5)
#     Y = model.action(X, 5, M)
#     print(Y)
## modify env configurations
adv_dir = "advs/Nov-02_14_53_46-5"
env.config["traffic_density"] = 4
env.config["adv_model"] = load(adv_dir)
env.config["duration"] = 20
env.config["reward_type"] = "collision"
models = []
for adv_m in adv_dir.split(","):
    models.append(load(adv_m))
env.config["adv_model"] = models

env.config["train_strat"] ="homo"
env.config["param"] = 0
env.config["ratio"] = 1


action_mask = None
crashed = 0
for i in range(runs):
    obs, mask = env.reset(is_training=True)
    action_mask = mask
    env.render()
    done = False
    while not done:
        act = cav_model.action(obs, 3 ,mask)
        # act = [0] * 2
        obs, reward, done, info = env.step(act)
        mask = info["action_mask"]
        # regional_rewards
        avg_rew = round(sum(info["regional_rewards"])/len(info["regional_rewards"]),2)
        env.render()
    crash = any(vehicle.crashed for vehicle in env.controlled_vehicles)
    if crash:
        print("crashed")

    # print("reg_rew:" + str(avg_rew) + "   only Cavs: "+ str(env.only_cavs_crashes()))
