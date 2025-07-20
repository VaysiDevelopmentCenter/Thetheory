import unittest
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.engine.network_graph import NetworkGraph
from modules.evaluation.evaluation_oracle import EvaluationOracle
from modules.evaluation.reward import Reward

class TestEvaluation(unittest.TestCase):

    def test_evaluation_oracle(self):
        """Test the EvaluationOracle."""
        # Create a simple NetworkGraph
        graph = NetworkGraph()
        input_layer = graph.add_layer('input', {})
        hidden_layer = graph.add_layer('dense', {'units': 128})
        output_layer = graph.add_layer('dense', {'units': 10})
        graph.add_connection(input_layer, hidden_layer)
        graph.add_connection(hidden_layer, output_layer)

        # Create an EvaluationOracle
        oracle = EvaluationOracle('cifar10', 'classification')

        # Evaluate the graph
        metrics = oracle.evaluate(graph)

        # Check the metrics
        self.assertIn('accuracy', metrics)
        self.assertIn('loss', metrics)
        self.assertIn('size', metrics)
        self.assertEqual(metrics['accuracy'], 0.95)
        self.assertEqual(metrics['loss'], 0.1)
        self.assertEqual(metrics['size'], 3)

    def test_reward_calculation(self):
        """Test the Reward class."""
        weights = {'accuracy': 1.0, 'size': 0.001}
        reward_calculator = Reward(weights)
        metrics = {'accuracy': 0.95, 'loss': 0.1, 'size': 100}
        reward = reward_calculator.calculate(metrics)
        self.assertAlmostEqual(reward, 0.95 - 0.1)

if __name__ == '__main__':
    unittest.main()
