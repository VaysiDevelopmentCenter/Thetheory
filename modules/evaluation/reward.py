class Reward:
    """
    Calculates a reward based on performance metrics.
    """
    def __init__(self, weights):
        """
        :param weights: A dictionary of weights for each metric.
        """
        self.weights = weights

    def calculate(self, metrics):
        """
        Calculates the reward for the given metrics.

        :param metrics: A dictionary of performance metrics.
        :return: The calculated reward.
        """
        reward = 0
        for metric, weight in self.weights.items():
            if metric in metrics:
                # We want to maximize accuracy and minimize size, so we subtract size.
                if metric == 'size':
                    reward -= metrics[metric] * weight
                else:
                    reward += metrics[metric] * weight
        return reward
