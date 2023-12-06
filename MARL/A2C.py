import torch as th
import os, logging
import configparser


# config_dir = 'configs/configs.ini'
# config = configparser.ConfigParser()
# config.read(config_dir)
# torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')
# th.manual_seed(torch_seed)
# th.backends.cudnn.benchmark = False
# th.backends.cudnn.deterministic = True
# os.environ['PYTHONHASHSEED'] = str(torch_seed)

import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np

from common.Agent import Agent
from common.Model import ActorNetwork, CriticNetwork
from common.utils import entropy, index_to_one_hot, to_tensor_var, VideoRecorder


class A2C(Agent):
    """
    An agent learned with Advantage Actor-Critic
    - Actor takes state as input
    - Critic takes both state and action as input
    - agent interact with environment to collect experience
    - agent training with experience to update policy
    """
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=10,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                  critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True,
                 traffic_density=1, test_seeds=0, state_split=False, reward_type='regionalR'):
        super(A2C, self).__init__(env, state_dim, action_dim,
                 memory_capacity, max_steps,
                 reward_gamma, reward_scale, done_penalty,
                 actor_hidden_size, critic_hidden_size, critic_loss,
                 actor_lr, critic_lr,
                 optimizer_type, entropy_reg,
                 max_grad_norm, batch_size, episodes_before_train,
                 epsilon_start, epsilon_end, epsilon_decay,
                 use_cuda)

        assert traffic_density in [1, 2, 3, 4, 5, 6]
        assert reward_type in ["collision", "merge", "global", "selfish"]
        self.roll_out_n_steps = roll_out_n_steps
        self.test_seeds = test_seeds
        self.traffic_density = traffic_density
        self.reward_type = reward_type


        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                  self.action_dim, state_split=True)
        self.critic = CriticNetwork(self.state_dim, self.critic_hidden_size, 
                                    1, state_split=True)
        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)
        if self.use_cuda:
            self.actor.cuda()

        self.episode_rewards = [0]
        self.average_speed = [0]
        self.epoch_steps = [0]


    # agent interact with the environment to collect experience
    def interact(self):
        self._take_n_steps()

    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass
        ## store initial actor weights
        initial_weights = self.actor.fc11.weight.data.clone()


        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
        action_masks_var = to_tensor_var(batch.action_masks, self.use_cuda).view(-1, self.action_dim)
        one_hot_actions = index_to_one_hot(batch.actions, self.action_dim)
        actions_var = to_tensor_var(one_hot_actions, self.use_cuda).view(-1, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)

        # update actor network
        self.actor_optimizer.zero_grad()
        action_log_probs = self.actor(states_var, action_masks_var)
        entropy_loss = th.mean(entropy(th.exp(action_log_probs)))
        # e_action_log_probs = th.exp(action_log_probs)
        # p_action_log_probs = th.log(e_action_log_probs)
        # x = e_action_log_probs * p_action_log_probs
        # entropy_loss = th.mean(-th.sum(x,1))
        action_log_probs = th.sum(action_log_probs * actions_var, 1)
        values = self.critic(states_var)
        advantages = rewards_var - values.detach()
        pg_loss = -th.mean(action_log_probs * advantages)
        actor_loss = pg_loss - entropy_loss * self.entropy_reg
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic network
        self.critic_optimizer.zero_grad()
        target_values = rewards_var
        if self.critic_loss == "huber":
            critic_loss = nn.functional.smooth_l1_loss(values, target_values)
        else:
            critic_loss = nn.MSELoss()(values, target_values)

        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        # Check if weights have changed
        # weights_changed = not initial_weights.equal(self.actor.fc11.weight.data)
        # weights_changed = not nn.equal(initial_weights, self.actor.fc1.weight.data)
        # print("n_steps %d changed %d" % (self.n_steps, weights_changed))

    # predict softmax action based on state
    def _softmax_action(self, state, action_mask):
        state_var = to_tensor_var([state], self.use_cuda)
        action_mask_var = to_tensor_var([action_mask], self.use_cuda)
        softmax_action_var = th.exp(self.actor(state_var, action_mask_var))
        if self.use_cuda:
            softmax_action = softmax_action_var.data.cpu().numpy()[0]
        else:
            softmax_action = softmax_action_var.data.numpy()[0]
        return softmax_action

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state, action_mask):
        softmax_action = self._softmax_action(state, action_mask)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                  np.exp(-1. * self.n_steps / self.epsilon_decay)
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = np.argmax(softmax_action)
        return action, softmax_action

    # choose an action based on state for execution
    def action(self, state, action_mask):
        softmax_action = self._softmax_action(state, action_mask)
        action = np.argmax(softmax_action)
        return action

    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action = index_to_one_hot(action, self.action_dim)
        action_var = to_tensor_var([action], self.use_cuda)
        value_var = self.critic(state_var)
        if self.use_cuda:
            value = value_var.data.cpu().numpy()[0]
        else:
            value = value_var.data.numpy()[0]
        return value

    # evaluation the learned agent
    def evaluation(self, env, output_dir, eval_episodes=1, is_train=True, adv_type="coll_rew", render = False):
        global_rewards = []
        reg_rewards = []
        reg_wadv_rewards = []
        infos = []
        avg_speeds = []
        steps = []
        vehicle_speed = []
        vehicle_position = []
        video_recorder = None
        seeds = [int(s) for s in self.test_seeds.split(',')]
        adv_rewards = []

        for i in range(eval_episodes):
            done = False
            avg_speed = 0
            step = 0
            global_i = []
            reg_i = []
            reg_wadv_i = []
            infos_i = []
            adv_rewards_i = []
            if is_train:
                if self.traffic_density == 1:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 1)
                elif self.traffic_density == 2:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 2)
                elif self.traffic_density == 3:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 4)
                elif self.traffic_density == 4:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV = 5)
                elif self.traffic_density == 5:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV = 8)
                elif self.traffic_density == 6:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV = 10)
            else:
                state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i])

            n_agents = len(env.controlled_vehicles)
            if render:
                rendered_frame = env.render(mode="rgb_array")
                video_filename = os.path.join(output_dir,
                                            "testing_episode{}".format(self.n_episodes + 1) + '_{}'.format(i) +
                                            '.mp4')
                # Init video recording
                if video_filename is not None:
                    print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape,
                                                                        5))
                    video_recorder = VideoRecorder(video_filename,
                                                frame_size=rendered_frame.shape, fps=5)
                    video_recorder.add_frame(rendered_frame)
                else:
                    video_recorder = None

            while not done:
                step += 1
                action = self.action(state, action_mask)
                state, global_reward, done, info = env.step(action)
            
                
                action_mask = info["action_mask"]
                avg_speed += info["average_speed"]
                rendered_frame = env.render(mode="rgb_array")
                if render and video_recorder is not None:
                    video_recorder.add_frame(rendered_frame)

                ## data collected used for evaluation
                reg = info["regional_reward"]
                reg_w = info["regional_reward_wadv"]

                global_i.append(global_reward)
                reg_i.append(reg)
                reg_wadv_i.append(reg_w)

                infos_i.append(info)
                adv_rewards_i.append(info[adv_type])

            vehicle_speed.append(info["vehicle_speed"])
            vehicle_position.append(info["vehicle_position"])

            global_rewards.append(global_i)
            reg_rewards.append(reg_i)
            reg_wadv_rewards.append(reg_wadv_i)


            adv_rewards.append(adv_rewards_i)
            infos.append(infos_i)
            steps.append(step)
            avg_speeds.append(avg_speed / step)

        if render and  video_recorder is not None:
            video_recorder.release()
        env.close()
        return adv_rewards, global_rewards, reg_rewards, reg_wadv_rewards, (vehicle_speed, vehicle_position), steps, avg_speeds

    def load(self, model_dir, global_step=None, train_mode=False):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if global_step is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        tokens = file.split('.')[0].split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = file
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)
        if save_file is not None:
            file_path = model_dir + save_file
            checkpoint = th.load(file_path)
            print('Checkpoint loaded: {}'.format(file_path))
            # logging.info('Checkpoint loaded: {}'.format(file_path))
            self.actor.load_state_dict(checkpoint['model_state_dict'])
            if train_mode:
                self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.actor.train()
            else:
                self.actor.eval()
            
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False

    def save(self, model_dir, global_step):        
        file_path = model_dir + 'checkpoint-{:d}.pt'.format(global_step)
        # if not self.shared_network:
        th.save({'global_step': global_step,
                    'model_state_dict': self.actor.state_dict(),
                    'optimizer_state_dict': self.actor_optimizer.state_dict()},
                file_path)
        # else:
        #     th.save({'global_step': global_step,
        #              'model_state_dict': self.policy.state_dict(),
        #              'optimizer_state_dict': self.policy_optimizers.state_dict()},
        #             file_path)
            
    # take one step
    def _take_one_step(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state = self.env.reset()
            self.n_steps = 0
        state = self.env_state
        action = self.exploration_action(self.env_state)
        next_state, reward, done, _ = self.env.step(action)
        if done:
            if self.done_penalty is not None:
                reward = self.done_penalty
            next_state = [0] * len(state)
            self.env_state = self.env.reset()
            self.n_episodes += 1
            self.episode_done = True
        else:
            self.env_state = next_state
            self.episode_done = False
        self.n_steps += 1
        self.memory.push(state, action, reward, next_state, done)

    # take n steps
    def _take_n_steps(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state, self.action_mask = self.env.reset()
            self.n_steps = 0
        states = []
        actions = []
        rewards = []
        policies = []
        action_masks = []
        done = True
        average_speed = []
        # take n steps
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action_masks.append(self.action_mask)
            action, policy = self.exploration_action(self.env_state, self.action_mask)
            next_state, global_reward, done, info = self.env.step(action)

            self.episode_rewards[-1] += global_reward
            self.epoch_steps[-1] += 1
            
            if self.reward_type == "collision":
                reward = info["coll_rew"]
            elif self.reward_type == "merge":
                reward = info["merge_rew"]
            elif self.reward_type == "global":
                reward = info["global_rew"]
            elif self.reward_type == "selfish":
                reward = info["selfish_rew"]

            average_speed += info["average_speed"]
            actions.append(action)
            rewards.append(reward)
            policies.append(policy)
            final_state = next_state

            self.env_state = next_state
            self.action_mask = info['action_mask']

            self.n_steps += 1
            if done and self.done_penalty is not None:
                reward = self.done_penalty

            if done:
                self.env_state, self.action_mask = self.env.reset()
                break
        # discount reward
        if done:
            final_value = 0.0
            self.n_episodes += 1
            self.episode_done = True
            self.average_speed[-1] = average_speed / self.epoch_steps[-1]
            self.episode_rewards.append(0)
            self.epoch_steps.append(0)
            self.average_speed.append(0)
        else:
            self.episode_done = False
            final_action = self.action(final_state, self.action_mask)
            final_value = self.value(final_state, final_action)

        rewards = self._discount_reward(rewards, final_value)
        if self.reward_scale > 0:
            rewards = np.array(rewards) / self.reward_scale
        self.n_steps += 1
        self.memory.push(states, actions, rewards, policies, action_masks)

    # discount roll out rewards
    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    # soft update the actor target network or critic target network
    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)