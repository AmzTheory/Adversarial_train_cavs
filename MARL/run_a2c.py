from __future__ import print_function, division
from A2C import A2C
from common.utils import agg_double_list, copy_file, init_dir
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


import argparse
import configparser
import sys
sys.path.append("../highway-env")

import gym
import os
import highway_env
import numpy as np
import matplotlib.pyplot as plt
import torch as th


def parse_args():
    """
    Description for this experiment:
        + hard: 7-steps, curriculum
    """
    default_base_dir = "./results/"
    default_config_dir = 'configs/configs.ini'
    parser = argparse.ArgumentParser(description=('Train or evaluate policy on RL environment '
                                                  'using MA2C'))
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--option', type=str, required=False,
                        default='train', help="train or evaluate")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    parser.add_argument('--model-dir', type=str, required=False,
                        default='', help="pretrained model path")
    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="random seeds for evaluation, split by ,")
    parser.add_argument('--adv', type=int, required=False,
                        default=0, help="provide the adversary type")
    args = parser.parse_args()
    return args


def train(args, seed = None):
    base_dir = args.base_dir
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    # create an experiment folder
    now = datetime.now().strftime("%b-%d_%H_%M_%S") + ("-"+str(seed) if seed else "")
    output_dir = base_dir + now
    dirs = init_dir(output_dir)
    copy_file(dirs['configs'])

    if os.path.exists(args.model_dir):
        model_dir = args.model_dir
    else:
        model_dir = dirs['models']

    torch_seed = seed if seed else config.getint('MODEL_CONFIG', 'torch_seed')
    th.manual_seed(torch_seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(torch_seed)

    # model configs
    BATCH_SIZE = config.getint('MODEL_CONFIG', 'BATCH_SIZE')
    MEMORY_CAPACITY = config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY')
    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM')
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'ENTROPY_REG')
    epsilon = config.getfloat('MODEL_CONFIG', 'epsilon')
    alpha = config.getfloat('MODEL_CONFIG', 'alpha')
    state_split = config.getboolean('MODEL_CONFIG', 'state_split')
    reward_type = config.get('MODEL_CONFIG', 'reward_type')
    eps_start = config.getfloat('MODEL_CONFIG', 'EPSILON_START')
    eps_end = config.getfloat('MODEL_CONFIG', 'EPSILON_END')
    eps_decay = config.getfloat('MODEL_CONFIG', 'EPSILON_DECAY')


    # train configs
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')
    MAX_EPISODES = config.getint('TRAIN_CONFIG', 'MAX_EPISODES')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    EVAL_INTERVAL = config.getint('TRAIN_CONFIG', 'EVAL_INTERVAL')
    EVAL_EPISODES = config.getint('TRAIN_CONFIG', 'EVAL_EPISODES')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')
    # init env
    env = gym.make('adv-merge-v1')
    env.config['seed'] = seed if seed else config.getint('ENV_CONFIG', 'seed')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['safety_guarantee'] = config.getboolean('ENV_CONFIG', 'safety_guarantee')
    env.config['n_step'] = config.getint('ENV_CONFIG', 'n_step')
    # env.config['cav_model'] = loadmodel(config.get('ENV_CONFIG', 'CAV_MODEL'))
    traffic_density = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')
    env.config['reward_type'] = reward_type
    assert env.T % ROLL_OUT_N_STEPS == 0

    env_eval = gym.make('adv-merge-v1')
    env_eval.config['seed'] = config.getint('ENV_CONFIG', 'seed') + 1
    env_eval.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env_eval.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env_eval.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env_eval.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env_eval.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env_eval.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env_eval.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env_eval.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env_eval.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    env_eval.config['safety_guarantee'] = config.getboolean('ENV_CONFIG', 'safety_guarantee')
    env_eval.config['n_step'] = config.getint('ENV_CONFIG', 'n_step')
    env_eval.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')
    # env_eval.config['cav_model'] = env.config['cav_model']
    env_eval.config['reward_type'] = reward_type    

    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.evaluation_seeds
    advs = ["coll_rew", "merge_rew", "global_rew", "selfish_rew"]
    adv_type = advs[args.adv]

    a2c = A2C(env, state_dim=state_dim, action_dim=action_dim,
                 memory_capacity=MEMORY_CAPACITY, max_steps=None,
                 roll_out_n_steps=ROLL_OUT_N_STEPS,
                 reward_gamma=reward_gamma, reward_scale=reward_scale, done_penalty=None,
                 actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                 actor_lr=actor_lr, critic_lr=critic_lr,
                 optimizer_type="rmsprop", entropy_reg=ENTROPY_REG,
                 max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
                 episodes_before_train=EPISODES_BEFORE_TRAIN,
                 use_cuda=False,
                 epsilon_start=eps_start, epsilon_end= eps_end, epsilon_decay=eps_decay, 
                 traffic_density=traffic_density, test_seeds=test_seeds,
                 state_split=state_split,  reward_type=reward_type)

    # load the model if exist
    a2c.load(model_dir, train_mode=True)
    
    env.seed = env.config['seed']
    env.unwrapped.seed = env.config['seed']
    episodes = []
    gl_rewards = []
    rg_rewards = []
    rgw_rewards = []
    adv_eval_rewards = []
    steps_eps = []

    best_eval_reward = -100000

    while a2c.n_episodes < MAX_EPISODES:
        a2c.interact()
        if a2c.n_episodes >= EPISODES_BEFORE_TRAIN:
            a2c.train()

        if a2c.episode_done and ((a2c.n_episodes + 1) % EVAL_INTERVAL == 0):
            adv_rewards, global_rewards, reg_rewards, reg_wadv_rewards, _, steps, _ = a2c.evaluation(env_eval, dirs['train_videos'], EVAL_EPISODES, adv_type=adv_type)
            gl_mu, gl_std = agg_double_list(global_rewards)
            rg_mu, rg_std = agg_double_list(reg_rewards)
            rgw_mu, rgw_std = agg_double_list(reg_wadv_rewards)
            success_rate = sum(np.array(steps) == 100) / len(steps)
            adv_rewards_mu, adv_rewards_std = agg_double_list(adv_rewards)

            print("Episode %d, Adv_Reward %.2f" % (a2c.n_episodes + 1, adv_rewards_mu))
            print("collision Rate %.2f" % (1 - success_rate))
            print("global %.2f, regional %.2f, regional_wadv %.2f" % (gl_mu, rg_mu, rgw_mu))
            
            ## store the data for future analysis
            episodes.append(a2c.n_episodes + 1)
            gl_rewards.append(global_rewards)
            rg_rewards.append(reg_rewards)
            rgw_rewards.append(reg_wadv_rewards)
            adv_eval_rewards.append(adv_rewards)
            steps_eps.append(steps)
            
            # save the model
            if adv_rewards_mu > best_eval_reward:
                a2c.save(dirs['models'], 100000)
                a2c.save(dirs['models'], a2c.n_episodes + 1)
                best_eval_reward = adv_rewards_mu
            else:
                a2c.save(dirs['models'], a2c.n_episodes + 1)


        ## save raw data for metrics
        np.save(output_dir + '/{}'.format('global_rewards'), np.array(gl_rewards))
        np.save(output_dir + '/{}'.format('reg_rewards'), np.array(rg_rewards))
        np.save(output_dir + '/{}'.format('reg_adv_rewards'), np.array(rgw_rewards))
        np.save(output_dir + '/{}'.format('adv_eval_rewards'), np.array(adv_eval_rewards))
        np.save(output_dir + '/{}'.format('steps'), np.array(steps_eps))
        
        # save other metrics
        np.save(output_dir + '/{}'.format('episode_rewards'), np.array(a2c.episode_rewards))
        np.save(output_dir + '/{}'.format('epoch_steps'), np.array(a2c.epoch_steps))
        np.save(output_dir + '/{}'.format('average_speed'), np.array(a2c.average_speed))

    # save the model
    a2c.save(dirs['models'], MAX_EPISODES + 2)

    # plt.figure()
    # plt.plot(adv_eval_rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Average Training adv Reward")
    # plt.legend(["A2C"])
    # plt.savefig(output_dir + '/' + "a2c_train.png")
    # plt.show()


def evaluate(args, seed = None):
    if os.path.exists(args.model_dir):
        model_dir = args.model_dir + '/models/'
    else:
        raise Exception("Sorry, no pretrained models")
    
    config_dir = args.model_dir + '/configs/configs.ini'
    config = configparser.ConfigParser()
    config.read(config_dir)

    video_dir = args.model_dir + '/eval_videos'
    eval_logs = args.model_dir + '/eval_logs'

    torch_seed = seed if seed else config.getint('MODEL_CONFIG', 'torch_seed')
    th.manual_seed(torch_seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(torch_seed)

    # model configs
    BATCH_SIZE = config.getint('MODEL_CONFIG', 'BATCH_SIZE')
    MEMORY_CAPACITY = config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY')
    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM')
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'ENTROPY_REG')
    epsilon = config.getfloat('MODEL_CONFIG', 'epsilon')
    alpha = config.getfloat('MODEL_CONFIG', 'alpha')
    state_split = config.getboolean('MODEL_CONFIG', 'state_split')
    reward_type = config.get('MODEL_CONFIG', 'reward_type')
    eps_start = config.getfloat('MODEL_CONFIG', 'EPSILON_START')
    eps_end = config.getfloat('MODEL_CONFIG', 'EPSILON_END')
    eps_decay = config.getfloat('MODEL_CONFIG', 'EPSILON_DECAY')

    # train configs
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')

    # init env
    env = gym.make('adv-merge-v1')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['safety_guarantee'] = config.getboolean('ENV_CONFIG', 'safety_guarantee')
    env.config['n_step'] = config.getint('ENV_CONFIG', 'n_step')
    traffic_density = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')
    # env.config['cav_model'] = loadmodel(config.get('ENV_CONFIG', 'CAV_MODEL'))
    env.config['reward_type'] = reward_type
    
    assert env.T % ROLL_OUT_N_STEPS == 0
    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.evaluation_seeds
    seeds = [int(s) for s in test_seeds.split(',')]
    advs = ["coll_rew", "merge_rew", "global_rew", "selfish_rew"]
    adv_type = advs[args.adv]

    a2c = A2C(env, state_dim=state_dim, action_dim=action_dim,
                 memory_capacity=MEMORY_CAPACITY, max_steps=None,
                 roll_out_n_steps=ROLL_OUT_N_STEPS,
                 reward_gamma=reward_gamma, reward_scale=reward_scale, done_penalty=None,
                 actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                 actor_lr=actor_lr, critic_lr=critic_lr,
                 optimizer_type="rmsprop", entropy_reg=ENTROPY_REG,
                 max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
                 episodes_before_train=EPISODES_BEFORE_TRAIN,
                 use_cuda=False,
                 epsilon_start=eps_start, epsilon_end= eps_end, epsilon_decay=eps_decay, 
                 traffic_density=traffic_density, test_seeds=test_seeds,
                 state_split=state_split,  reward_type=reward_type)
    

    # load the model if exist
    a2c.load(model_dir, train_mode=False)
    # adv_rewards, rewards, (vehicle_speed, vehicle_position), steps, avg_speeds = a2c.evaluation(env, video_dir, len(seeds), is_train=False,adv_type = adv_type)
    # rewards_mu, rewards_std = agg_double_list(rewards)
    # adv_rewards_mu, adv_rewards_std = agg_double_list(adv_rewards)
    # success_rate = sum(np.array(steps) == 100) / len(steps)
    

    adv_rewards, global_rewards, reg_rewards, reg_wadv_rewards, (vehicle_speed, vehicle_position), steps, avg_speeds = a2c.evaluation(env, video_dir, len(seeds), is_train=False, adv_type=adv_type)
    gl_mu, gl_std = agg_double_list(global_rewards)
    rg_mu, rg_std = agg_double_list(reg_rewards)
    rgw_mu, rgw_std = agg_double_list(reg_wadv_rewards)
    success_rate = sum(np.array(steps) == 100) / len(steps)
    adv_rewards_mu, adv_rewards_std = agg_double_list(adv_rewards)
    avg_speeds_mu, avg_speeds_std = agg_double_list(avg_speeds)

    # print("Evaluation Reward and std %.2f, %.2f " % (rewards_mu, rewards_std))
    # print("Adversary Reward and std %.2f, %.2f " % (adv_rewards_mu, adv_rewards_std))
    # print("Collision Rate %.2f" % (1 - success_rate))
    # print("Average Speed and std %.2f , %.2f " % (avg_speeds_mu, avg_speeds_std))


    np.save(eval_logs + '/{}'.format('eval_rewards'), np.array(global_rewards))
    np.save(eval_logs + '/{}'.format('adv_eval_rewards'), np.array(adv_rewards))
    np.save(eval_logs + '/{}'.format('eval_steps'), np.array(steps))
    np.save(eval_logs + '/{}'.format('eval_avg_speeds'), np.array(avg_speeds))
    np.save(eval_logs + '/{}'.format('vehicle_speed'), np.array(vehicle_speed))
    np.save(eval_logs + '/{}'.format('vehicle_position'), np.array(vehicle_position))

    return avg_speeds_mu, (1-success_rate), np.mean(np.array(steps))


def load(dir, checkpoint=None):
    if os.path.exists(dir):
        model_dir = dir + '/models/'
    else:
        raise Exception("Sorry, no pretrained models")
    config_dir = dir + '/configs/configs.ini'
    config = configparser.ConfigParser()
    config.read(config_dir)


    torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')
    th.manual_seed(torch_seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(torch_seed)

    # model configs
    BATCH_SIZE = config.getint('MODEL_CONFIG', 'BATCH_SIZE')
    MEMORY_CAPACITY = config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY')
    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM')
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'ENTROPY_REG')
    epsilon = config.getfloat('MODEL_CONFIG', 'epsilon')
    alpha = config.getfloat('MODEL_CONFIG', 'alpha')
    state_split = config.getboolean('MODEL_CONFIG', 'state_split')
    # reward_type = config.get('MODEL_CONFIG', 'reward_type')
    reward_type = "collision"
    # eps_start = config.getfloat('MODEL_CONFIG', 'EPSILON_START')
    # eps_end = config.getfloat('MODEL_CONFIG', 'EPSILON_END')
    # eps_decay = config.getfloat('MODEL_CONFIG', 'EPSILON_DECAY')

    # train configs
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')

    # init env
    env = gym.make('merge-v1')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['safety_guarantee'] = config.getboolean('ENV_CONFIG', 'safety_guarantee')
    env.config['n_step'] = config.getint('ENV_CONFIG', 'n_step')
    traffic_density = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')
    # env.config['cav_model'] = loadmodel(config.get('ENV_CONFIG', 'CAV_MODEL'))
    env.config['reward_type'] = reward_type
    
    assert env.T % ROLL_OUT_N_STEPS == 0
    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = "2,3,4"
    seeds = [int(s) for s in test_seeds.split(',')]

    a2c = A2C(env, state_dim=state_dim, action_dim=action_dim,
                 memory_capacity=MEMORY_CAPACITY, max_steps=None,
                 roll_out_n_steps=ROLL_OUT_N_STEPS,
                 reward_gamma=reward_gamma, reward_scale=reward_scale, done_penalty=None,
                 actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                 actor_lr=actor_lr, critic_lr=critic_lr,
                 optimizer_type="rmsprop", entropy_reg=ENTROPY_REG,
                 max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
                 episodes_before_train=EPISODES_BEFORE_TRAIN,
                 use_cuda=False,
                 epsilon_start=0, epsilon_end= 0, epsilon_decay=0, 
                 traffic_density=traffic_density, test_seeds=test_seeds,
                 state_split=state_split,  reward_type=reward_type)
    
    a2c.load(model_dir, train_mode=False, global_step=checkpoint)
    
    return a2c


if __name__ == "__main__":
    args = parse_args()
    # train or eval
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
