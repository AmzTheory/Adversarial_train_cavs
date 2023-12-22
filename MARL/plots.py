
import argparse, glob, os,sys
import numpy as np
import run_ma2c

import matplotlib.pyplot as plt
from common.utils import agg_double_list

def parse_args():
    parser = argparse.ArgumentParser(description=('Visualize Policies'))
    parser.add_argument('--base-dir', type=str, required=False,
                        default="", help="experiment base dir")
    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="random seeds for evaluation, split by ,")
    parser.add_argument('--adv', type=int, required=False,
                        default=0, help="provide the adversary type")
    parser.add_argument('--option', type=str, required=False,
                        default="plot", help="select type")
    parser.add_argument('--y', type=str, required=False,
                        default="rew", help="select reward")
    parser.add_argument('--ratio', type=int, required=False,
                        default=1, help="select reward")
    parser.add_argument('--env-id', type=int, required=False,
                        default=0, help="select env id")
    parser.add_argument('--read-id', type=int, required=False,
                        default=0, help="select read id")
    parser.add_argument('--checkout', type=int, required=False,
                        default=None, help="checkout to evaluate")
    parser.add_argument('--save-path', type=str, required=False,
                        default=None, help="specify the path where the figure gets stored")

    args = parser.parse_args()
    return args

def evalaute_models(args, eval):

    rews, reg_rews, colls, avg_speeds= [], [], [], []

    for fp in glob.glob(os.path.join(args.base_dir, "*")):
        #ensure args is compatible with evaluate
        args.model_dir = fp
        rew, reg_rew, col, speed  = eval(args, gl = args.checkout)
        avg_speeds.append(speed)
        colls.append(col)
        rews.append(rew)
        reg_rews.append(reg_rew)

    a_rew, std_rew= np.mean(np.array(rews)), np.std(np.array(rews))
    a_rew_reg, std_rew_reg= np.mean(np.array(reg_rews)), np.std(np.array(reg_rews))
    a_coll, std_coll = np.mean(np.array(colls)), np.std(np.array(colls))
    a_speed, std_speed = np.mean(np.array(avg_speeds)), np.std(np.array(avg_speeds))


    print("Global Reward:     avg = %.2f std = %.2f " % (a_rew, std_rew) )
    print("Reg    Reward:     avg = %.2f std = %.2f " % (a_rew_reg, std_rew_reg) )
    print("Collision Rate:    avg = %.2f std = %.2f " % (a_coll, std_coll) )
    print("Speed:             avg = %.2f std = %.2f " % (a_speed, std_speed) )


def read(path, env_id = 0):
    rewards = np.load(path+"/rewards.npy", allow_pickle=True)
    rew_rewards = np.load(path+"/reg_rewards.npy", allow_pickle=True)
    steps = np.load(path+"/steps.npy", allow_pickle=True)
    speeds = np.load(path+"/speed.npy", allow_pickle=True)
    
    size = 100
    return rewards[:size], rew_rewards[:size], steps[:size], speeds[:size]

def read_ext(path, env_t = 0):
    if env_t == 0:
        rewards = np.load(path+"/rewards_coll.npy", allow_pickle=True)
        rew_rewards = np.load(path+"/reg_rewards_coll.npy", allow_pickle=True)
        steps = np.load(path+"/steps_coll.npy", allow_pickle=True)
        speeds = np.load(path+"/speed_coll.npy", allow_pickle=True)
        speeds = np.load(path+"/speed_coll.npy", allow_pickle=True)
    else:
        rewards = np.load(path+"/rewards_sp.npy", allow_pickle=True)
        rew_rewards = np.load(path+"/reg_rewards_sp.npy", allow_pickle=True)
        steps = np.load(path+"/steps_sp.npy", allow_pickle=True)
        speeds = np.load(path+"/speed_sp.npy", allow_pickle=True)
    
    size = 200
    return rewards[:size], rew_rewards[:size], steps[:size], speeds[:size]

def get_smoothed(data, w = 5):
    smoothed_data = np.empty_like(data, dtype=float)

# Calculate the rolling average
    for i in range(len(data)):
        start_index = max(0, i - w + 1)
        end_index = i + 1
        smoothed_data[i] = np.mean(data[start_index:end_index])

    return smoothed_data

def plot_models(args, models, met = "rew"):
    read_f = read_data if  args.read_id == 0 else read_data_ext
    for m, path in models.items():
        plot_data(path, met, label=m, smooth=True, read_func = read_f, env_id = args.env_id)

def read_data(path, env_id):
    rewards_l, rewards_reg_l, steps_l, speeds_l= read(path, env_id)
    mets = ["rew", "reg_rew", "avg_speed", "collision"]
    data = dict()
    for m in mets:
        data[m] = []
    i = 1
    for rewards, reg_rewards, steps, speeds in zip(rewards_l, rewards_reg_l, steps_l, speeds_l):
    
        rewards_mu, rewards_std = agg_double_list(rewards)
        rewards_reg_mu, rewards_reg_std = agg_double_list(reg_rewards)
        success_rate = sum(np.array(steps) == 100) / len(steps)

        # eps_mean = [np.mean(np.array(s_i)) for s_i in speeds]
        avg_speeds_mu, avg_speeds_std = agg_double_list(speeds)

        data["rew"].append(rewards_mu)
        data["reg_rew"].append(rewards_reg_mu)
        data["avg_speed"].append(avg_speeds_mu)
        data["collision"].append(1.0 - success_rate)

        # print("Episode %d, Adv_Reward %.2f" % (i, adv_rewards_mu))
        # print("collision Rate %.2f" % (1 - success_rate))
        # print("global %.2f, regional %.2f, regional_wadv %.2f" % (gl_mu, rg_mu, rgw_mu))
        # print("########################################")
        # i+=1
    return data


def plot_data(base_path, met, label, p = plt, smooth = False, read_func = read_data, env_id = 0):
    # assert met in ["rew", "reg_rew","avg_speed", "collision"]
    data = []
    for fp in glob.glob(os.path.join(base_path, "*")):
        data.append(read_func(fp, env_id))

    ## collect the corresponding metric for each run
    met_data = np.array([ np.array(d[met]) for d in data ])

    
    mean = np.mean(met_data, axis = 0)
    
    if smooth:
        window_size = 15
        # mean = np.convolve(mean, np.ones(window_size) / window_size, mode='same')
        mean = get_smoothed(mean, w = window_size)

    std = np.std(met_data, axis = 0)
    min_y = mean - std
    max_y = mean + std
    x = range(0, mean.size)
    

    p.plot(mean, label = label)
    # p.set_xlim(xmin = 1, xmax = mean.size)
    p.set_xticks([1] + list(range(25, mean.size-1, 50))) 
    p.tick_params(axis='both', which='major', labelsize=13)
    # plt.fill_between(x, mean, min_y, alpha = 0.5, linewidth = 1, color = "black")
    # plt.fill_between(x, mean, max_y, alpha = 0.5, linewidth = 1, color = "black")


def read_data_ext(path, env_id):
    rewards_l, rewards_reg_l, steps_l, speeds_l= read_ext(path, env_id)
    mets = ["rew", "reg_rew", "avg_speed", "collision"]
    data = dict()
    for m in mets:
        data[m] = []
    i = 1
    for rewards, reg_rewards, steps, speeds in zip(rewards_l, rewards_reg_l, steps_l, speeds_l):
    
        rewards_mu, rewards_std = agg_double_list(rewards)
        rewards_reg_mu, rewards_reg_std = agg_double_list(reg_rewards)
        success_rate = sum(np.array(steps) == 100) / len(steps)

        # eps_mean = [np.mean(np.array(s_i)) for s_i in speeds]
        avg_speeds_mu, avg_speeds_std = agg_double_list(speeds)

        data["rew"].append(rewards_mu)
        data["reg_rew"].append(rewards_reg_mu)
        data["avg_speed"].append(avg_speeds_mu)
        data["collision"].append(1.0 - success_rate)
    return data

def read_data_avg(path, env_id):
    rewards_l, rewards_reg_l, steps_l, speeds_l= read_ext(path, env_id)
    mets = ["rew_coll", "rew_sp", "rew_avg"]
    data = dict()
    
    rewards_coll_l = rewards = np.load(path+"/rewards_coll.npy", allow_pickle=True)
    rewards_sp_l = rewards = np.load(path+"/rewards_sp.npy", allow_pickle=True)

    for m in mets:
        data[m] = []
    i = 1
    for rewards_coll, rewards_sp in zip(rewards_coll_l, rewards_sp_l):
        rewards_mu_coll, _ = agg_double_list(rewards_coll)
        rewards_mu_sp, _ = agg_double_list(rewards_sp)

        rewards_mu = (rewards_mu_coll + rewards_mu_sp) / 2

        data["rew_coll"].append(rewards_mu_coll)
        data["rew_sp"].append(rewards_mu_sp)
        data["rew_avg"].append(rewards_mu)

    return data

def get_models(inp):
    models = dict()
    elements = [ elem for elem in  inp.split(",")]
    for elem in elements:
        values = elem.split(":")
        key = values[0]
        path = values[1]
        models[key] = path
    return models

def plot_fig(args, dets, output, read_f):
    # read_f = read_data if  args.read_id == 0 else read_data_ext
    models = get_models(args.base_dir)

    ## data reading
    # coll_data = read_data(coll_model, split=True)
    # sp_data = read_data(sp_model)


    fig, axes = plt.subplots(nrows=1, ncols=len(dets.keys()), figsize=(16, 4))

    ## plot each meteric in the corresponding axis
    for met, pl in zip(dets.keys(), axes):
        # plot_data(base_path=coll_model, met = met, smooth=True, p = pl, split=True,label="coll-adv")
        for key, m in models.items():
            plot_data(base_path=m, met = met, smooth=True, p = pl, label="$"+key+"$", read_func=read_f, env_id = args.env_id)
            # pl.set_title(dets[met][0])
            pl.set_xlabel("Evaluation Epochs", fontsize=14)
            pl.set_ylabel(dets[met][0], fontsize=14)
            pl.legend(fontsize=13)
    # Adjust the layout so that titles and labels do not overlap
    plt.tight_layout()

    # Display the figure
    if output:
        plt.savefig(output, dpi = 1000, format="pdf", bbox_inches = 'tight')
    else:
        plt.show()

def get_read(args):
    read_f = read_data
    if args.read_id == 1:
        read_f = read_data_ext
    elif args.read_id == 2:
        read_f = read_data_avg
    return read_f

if __name__ == "__main__":
    args = parse_args()
    if args.option == "eval":
        evalaute_models(args, run_ma2c.evaluate)
    elif args.option == "plot":
        read_f = read_data if  args.read_id == 0 else read_data_ext 
        plot_data(args.base_dir, args.y, "sol", smooth=True, read_func= read_f, env_id=args.env_id)
        plt.title(args.y)
        plt.xlabel("Evaluation epochs")
        plt.ylabel(args.y)
        plt.legend()
        plt.show()
    elif args.option == "plot_model":
        models = get_models(args.base_dir)
        plot_models(args, models, args.y)
        plt.title(args.y)
        plt.xlabel("Evaluation epochs")
        plt.ylabel(args.y)
        plt.legend()
        plt.show()
    elif args.option == "plot_fig":
        mets = {
            "rew":["Cumulative Global Reward "],
            "reg_rew":["Cumulative Regional Reward "],
            "collision":["Collision Rate (%)"],
            "avg_speed":["Average Speed (m/s)"],
        }

        plot_fig(args, mets, output=args.save_path, read_f = get_read(args) )

    elif args.option == "plot_fig_1":
        ["rew_coll", "rew_sp", "rew_avg"]
        mets = {
            "rew_coll":["Global Reward (adv_c)"],
            "rew_sp":["Global Reward (adv_s)"],
            "rew_avg":["Collision Rate"],
        }

        plot_fig(args, mets, output=args.save_path, read_f = get_read(args) )