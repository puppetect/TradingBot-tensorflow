from stable_baselines.deepq.policies import FeedForwardPolicy


class FFPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(FFPolicy, self).__init__(*args, **kwargs, layers=[128, 64, 32], feature_extraction='mlp')
