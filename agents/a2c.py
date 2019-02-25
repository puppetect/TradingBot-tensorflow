
import numpy as np
import tensorflow as tf
from stable_baselines import deepq
from stable_baselines.common import tf_util, TensorboardWriter
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.a2c.utils import find_trainable_variables
from stable_baselines import A2C


class TradingA2C(A2C):

    def __init__(self, policy, env, gamma=0.9, n_steps=5, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5, learning_rate=7e-4, alpha=0.99, epsilon=1e-5, lr_schedule='linear', tensorboard_log=None, _init_setup_model=True):

        super().__init__(policy=policy, env=env, gamma=gamma, n_steps=n_steps, vf_coef=vf_coef, ent_coef=ent_coef, max_grad_norm=max_grad_norm, learning_rate=learning_rate, alpha=alpha, epsilon=epsilon, lr_schedule=lr_schedule, tensorboard_log=tensorboard_log, _init_setup_model=_init_setup_model)

    def setup_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf_util.make_session(graph=self.graph)

            # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/deepq/build_graph.py
            step_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1, n_batch_step, reuse=False)
            self.params = find_trainable_variables('deepq')

            tf_util.initialize(self.sess)
            self.update_target(sess=self.sess)
            self.summary = tf.summary.merge_all()

    def learn(self, total_timesteps, seed=None, tb_log_name='DQN', test_interval=1, reset_num_timesteps=True):
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

            best_train_score = None
            best_test_score = None
            self.reward_curve = []

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

                if done:

                    print('-------------------------------------')
                    print('steps                     | {}'.format(self.num_timesteps))
                    print('episodes                  | {}'.format(len(episode_rewards)))
                    epsilon = int(100 * self.exploration.value(self.num_timesteps))
                    print('% time spent exploring    | {}'.format(epsilon))
                    print('--')

                    mean_100ep_reward = -np.inf if len(episode_rewards[-16:-1]) == 0 else round(float(np.mean(episode_rewards[-16:-1])), 1)
                    self.reward_curve.append(mean_100ep_reward)
                    print('mean 10 episode reward    | {:.1f}'.format(mean_100ep_reward))

                    journal = self.env.sim.journal
                    print('Total operations          | {}'.format(len(self.env.sim.journal)))
                    longs = [x for x in journal if x['Type'] == 'LONG']
                    shorts = [x for x in journal if x['Type'] == 'SHORT']
                    print('Long/Short                | {}/{}'.format(len(longs), len(shorts)))
                    print('Avg duration trades       | {:.2f}'.format(np.mean([j['Trade Duration'] for j in journal])))
                    total_profit = sum([j['Profit'] for j in journal])
                    print('Total profit              | {:.2f}'.format(total_profit))
                    print('Avg profit per trade      | {:.3f}'.format(total_profit / self.env.sim.total_trades))

                    if epsilon <= self.exploration_final_eps * 100:
                        if best_train_score is None or total_profit > best_train_score:
                            self.save('saves/best_model_train.pkl')
                            best_train_score = total_profit

                    if self.num_timesteps % test_interval == 0:
                        print('--')
                        test_episode_rewards, test_longs, test_shorts, test_ave_profit_per_trade = self.test()
                        print('Total profit test         > {:.2f}'.format(test_episode_rewards))
                        print('Long/Short test           > {}/{}'.format(test_longs, test_shorts))
                        print('Avg profit per trade test > {:.3f}'.format(test_ave_profit_per_trade))

                        if epsilon <= self.exploration_final_eps * 100:
                            if best_test_score is None or test_episode_rewards > best_test_score:
                                self.save('saves/best_model_test.pkl')
                                best_test_score = test_episode_rewards
                    print('-------------------------------------')

                    obs = self.env.reset()
                    episode_rewards.append(0.0)

                    if self.num_timesteps + (self.num_timesteps / len(episode_rewards)) >= total_timesteps:
                        self.save('saves/final_model.pkl')
                        break

                self.num_timesteps += 1
        return self

    def test(self):
        obs = self.env.reset(train=False)
        done = False
        while not done:
            action, _ = self.predict(obs)
            obs, reward, done, info = self.env.step(action)
            journal = self.env.sim.journal
            longs = len([x for x in journal if x['Type'] == 'LONG'])
            shorts = len([x for x in journal if x['Type'] == 'SHORT'])
            test_episode_rewards = sum([j['Profit'] for j in journal])
            test_ave_profit_per_trade = test_episode_rewards / self.env.sim.total_trades if self.env.sim.total_trades > 0 else -np.inf
        return test_episode_rewards, longs, shorts, test_ave_profit_per_trade

    def save(self, save_path):
        data = {
            'batch_size': self.batch_size,
            'learning_starts': self.learning_starts,
            'learning_rate': self.learning_rate,
            'target_network_update_freq': self.target_network_update_freq,
            'exploration_final_eps': self.exploration_final_eps,
            'exploration_fraction': self.exploration_fraction,
            'gamma': self.gamma,
            'policy': self.policy,
            'journal': self.env.sim.journal,
            'reward_curve': self.reward_curve
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)
