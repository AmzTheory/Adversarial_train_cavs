from run_a2c import evaluate, load

from common.utils import agg_double_list, copy_file, init_dir
import glob, os
import numpy as np
import argparse
from run_ma2c import loadmodel

def gen_data(dir):
    seeds = [str(i) for i in range(0, 600, 100)]
    len_seeds = len(seeds)
    seeds = ','.join(seeds)


    speed_data = []
    gl_rewards = []
    rg_rewards = []
    rgw_rewards = []
    steps_eps = []
    speed_eps = []

    for p in [a for a in range(100, 6001, 100)]:
        print(p)
        model = loadmodel(dir, global_step = p)
        model.test_seeds = seeds
        env = model.env
        # cav_model = "MAA2C_5/Oct-02_12_53_20"
        # env.config['cav_model'] = loadmodel(cav_model, global_step=p)
        # avg_speeds = []
        # for s in seeds:
        # _, _, _, _, _, _, avg_speed, _, _ = model.evaluation(env, "", 10, is_train=False, render = False)
        global_rewards, reg_rewards, (vehicle_speed, vehicle_position), steps, avg_speeds, advs_crashes, cavs_crashes = model.evaluation(env, fp+"/eval_videos/", len_seeds, is_train=False, render = False)
        gl_mu, gl_std = agg_double_list(global_rewards)
        rg_mu, rg_std = agg_double_list(reg_rewards)
        # rgw_mu, rgw_std = agg_double_list(reg_wadv_rewards)
        success_rate = sum(np.array(steps) == 100) / len(steps)
        avg_speeds_mu, avg_speeds_std = agg_double_list(avg_speeds)

        adv_crashes_rate = np.mean(np.array(advs_crashes))
        cav_crashes_rate = np.mean(np.array(cavs_crashes))



        print("global  and std %.2f, %.2f " % (gl_mu, gl_std))
        print("reg     and std %.2f, %.2f " % (rg_mu, rg_std))
        print("Collision Rate %.2f adv rate %.2f cav rate %.2f" % (1 - success_rate, adv_crashes_rate, cav_crashes_rate))
        print("Average Speed and std %.2f , %.2f " % (avg_speeds_mu, avg_speeds_std))
        
        ## store the data for future analysis
        gl_rewards.append(global_rewards)
        rg_rewards.append(reg_rewards)
        # rgw_rewards.append(reg_wadv_rewards)
        steps_eps.append(steps)
        speed_eps.append(avg_speeds)


    ## save file
    ## save raw data for metrics
    dir = fp
    np.save(dir + '/{}'.format('global_rewards'), np.array(gl_rewards))
    np.save(dir + '/{}'.format('reg_rewards'), np.array(rg_rewards))
    np.save(dir + '/{}'.format('steps'), np.array(steps_eps))
    np.save(dir + '/{}'.format('eval_avgspeed'), np.array(speed_eps))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=True)
    args = parser.parse_args()
    return args

args = parse_args()
path = args.base_dir

for fp in glob.glob(os.path.join(path, "*")):
    gen_data(fp)