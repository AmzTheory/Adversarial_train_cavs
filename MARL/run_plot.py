import matplotlib.pyplot as plt
import numpy as np
import argparse
from common.utils import agg_double_list


parser = argparse.ArgumentParser(description=('plot model in eval logs'))
parser.add_argument('--dir', type=str, required=False)


args = parser.parse_args()


dir = args.dir
eval_logs = "/eval_logs"
speeds_path = dir + eval_logs +"/eval_avg_speeds.npy"
rewards_path = dir + eval_logs + "/eval_rewards.npy" 
steps_path = dir + eval_logs +"/eval_steps.npy"

avg_speeds = np.load(speeds_path, allow_pickle= True)
rewards = np.load(rewards_path, allow_pickle= True)
steps = np.load(steps_path, allow_pickle= True)

# rewards_mean, rewards_std = agg_double_list(full_reward)
# rewards_mean = [np.sum(ep) for ep in rewards]
rewards_mu, rewards_std = agg_double_list(rewards)
success_rate = sum(np.array(steps) == 100) / len(steps)
avg_speeds_mu, avg_speeds_std = agg_double_list(avg_speeds)

print("avg_speed : " + str(avg_speeds_mu))
print("coll rate : " + str(success_rate))
print("avg_reward : " + str(rewards_mu))


# plt.plot(speeds)
# plt.xlabel("Episode")
# plt.ylabel("Average Speed")
# plt.legend(["MAA2C"])
# plt.savefig(dir + eval_logs + '/' + "maa2c_eval_speed.png")

# plt.clf()

# plt.plot(rewards_mean)
# plt.xlabel("Episode")
# plt.ylabel("avg rewards")
# plt.legend(["MAA2C"])
# plt.savefig(dir + eval_logs + '/' + "maa2c_eval_reward.png")

# plt.clf()

# plt.plot(steps)
# plt.xlabel("Episode")
# plt.ylabel("steps")
# plt.legend(["MAA2C"])
# plt.savefig(dir + eval_logs + '/' + "maa2c_eval_steps.png")


