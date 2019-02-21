import os
import gym
import envs
import numpy as np
import tensorflow as tf
from stable_baselines import deepq
from stable_baselines.common import tf_util, TensorboardWriter
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.a2c.utils import find_trainable_variables
from stable_baselines import DQN


class FFPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(FFPolicy, self).__init__(*args, **kwargs, layers=[128, 64, 32], feature_extraction='mlp')


class TradingDQN(DQN):

    def __init__(self, policy, env, gamma=0.99, batch_size=32, buffer_size=50000, learning_starts=10000, learning_rate=5e-4, target_network_update_freq=1000, exploration_final_eps=0.1, exploration_fraction=0.2, tensorboard_log=None, _init_setup_model=True):

        super().__init__(policy=policy, env=env, gamma=gamma, batch_size=batch_size, buffer_size=buffer_size, learning_starts=learning_starts, learning_rate=learning_rate, target_network_update_freq=target_network_update_freq, exploration_final_eps=exploration_final_eps, exploration_fraction=exploration_fraction, tensorboard_log=tensorboard_log, _init_setup_model=_init_setup_model)

        # self.policy = policy
        # self.env = env
        # self.batch_size = batch_size
        # self.buffer_size = buffer_size
        # self.learning_starts = learning_starts
        # self.learning_rate = learning_rate
        # self.target_network_update_freq = target_network_update_freq
        # self.exploration_final_eps = exploration_final_eps
        # self.exploration_fraction = exploration_fraction
        # self.tensorboard_log = tensorboard_log
        # self.full_tensorboard_log = full_tensorboard_log
        # self.graph = None
        # self.sess = None
        # self.act = None
        # self.train = None
        # self.update_target = None
        # self.step_model = None
        # self.replay_buffer = None
        # self.exploration = None
        # self.episode_reward = None
        # if _init_setup_model:
        #     self.setup_model()

    def setup_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf_util.make_session(graph=self.graph)

            # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/deepq/build_graph.py
            self.act, self.train_step, self.update_target, self.step_model = deepq.build_train(
                q_func=self.policy,
                ob_space=self.env.observation_space,
                ac_space=self.env.action_space,
                optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate),
                gamma=self.gamma,
                grad_norm_clipping=10,
                sess=self.sess
            )
            self.params = find_trainable_variables("deepq")

            tf_util.initialize(self.sess)
            self.update_target(sess=self.sess)
            self.summary = tf.summary.merge_all()

    def learn(self, total_timesteps, seed=None, tb_log_name='DQN', reset_num_timesteps=True):
        if reset_num_timesteps:
            self.num_timesteps = 0

        with TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:
            self._setup_learn(seed)

            self.replay_buffer = ReplayBuffer(size=self.buffer_size)
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                              initial_p=1.0,
                                              final_p=self.exploration_final_eps)
            episode_rewards = [0.0]
            obs = self.env.reset(train=True)

            for _ in range(total_timesteps):
                update_eps = self.exploration.value(self.num_timesteps)
                with self.sess.as_default():
                    action = self.act(np.array(obs)[None], update_eps=update_eps)[0]
                new_obs, rew, done, _ = self.env.step(action)

                self.replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                episode_rewards[-1] += rew

                if self.num_timesteps > self.learning_starts:
                    obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                    weights = np.ones_like(rewards)
                    if writer is not None:
                        if (1 + self.num_timesteps) % 100 == 0:
                            summary, td_errors = self.train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights, sess=self.sess)
                            writer.add_summary(summary, self.num_timesteps)
                    else:
                        _, td_errors = self.train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights, sess=self.sess)

                if self.num_timesteps > self.learning_starts and self.num_timesteps % self.target_network_update_freq == 0:
                    self.update_target(sess=self.sess)

                mean_100ep_reward = -np.inf if len(episode_rewards[-101:-1]) == 0 else round(float(np.mean(episode_rewards[-101:-1])), 1)
                num_episodes = len(episode_rewards)

                if done:
                    print("-------------------------------------")
                    print("steps                     | {}".format(self.num_timesteps))
                    print("episodes                  | {}".format(num_episodes))
                    print("% time spent exploring    | {}".format(int(100 * self.exploration.value(self.num_timesteps))))

                    print("--")

                    print("mean 100 episode reward   | {:.1f}".format(mean_100ep_reward))
                    print("Total operations          | {}".format(len(self.env.sim.journal)))
                    print("Avg duration trades       | {:.2f}".format(np.mean([j["Trade Duration"] for j in self.env.sim.journal])))
                    print("Total profit              | {:.2f}".format(sum([j['Profit'] for j in self.env.sim.journal])))
                    print("Avg profit per trade      | {:.3f}".format(self.env.sim.average_profit_per_trade))

                    if num_episodes % 20 == 0:
                        print("--")
                        profit, ave_profit = self.test()
                        print("Total profit test         > {:.2f}".format(profit))
                        print("Avg profit per trade test > {:.3f}".format(ave_profit))
                    print("-------------------------------------")

                    obs = self.env.reset()
                    episode_rewards.append(0.0)

                self.num_timesteps += 1
        return self

    def test(self, episodes=1):
        obs = self.env.reset(train=False)
        test_episode_rewards = []
        test_ave_profit_per_trade = []

        for episode in range(episodes):
            done = False
            while not done:
                action, _, _ = self.step_model.step(np.array(obs)[None], deterministic=True)
                obs, reward, done, info = self.env.step(action)

            test_episode_rewards.append(sum([j['Profit'] for j in self.env.sim.journal]))
            test_ave_profit_per_trade.append(self.env.sim.average_profit_per_trade)
        return np.mean(test_episode_rewards), np.mean(test_ave_profit_per_trade)

    def save(self, save_path):
        data = {
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "learning_rate": self.learning_rate,
            "target_network_update_freq": self.target_network_update_freq,
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "gamma": self.gamma,
            "policy": self.policy,
            "journal": self.env.sim.journal
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)


if __name__ == '__main__':
    env = gym.make('trading-v0')
    model = TradingDQN(FFPolicy, env, tensorboard_log='saves/log')
    model.learn(total_timesteps=1000000)
    model.save('saves/model.pkl')
