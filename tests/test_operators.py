import unittest
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.engine.network_graph import NetworkGraph
from modules.operators.architectural_operators import *
from modules.operators.weight_operators import *
from modules.operators.hyperparameter_operators import *

class TestOperators(unittest.TestCase):

    def setUp(self):
        """Set up a new NetworkGraph for each test."""
        self.graph = NetworkGraph()

    def test_add_layer_operator(self):
        """Test the AddLayerOperator."""
        input_layer_id = self.graph.add_layer('input', {})
        op = AddLayerOperator('conv2d', {'filters': 32, 'kernel_size': 3}, input_layer_ids=[input_layer_id])
        new_layer_id = op.apply(self.graph)
        self.assertEqual(new_layer_id, 1)
        self.assertIn(new_layer_id, self.graph.graph.nodes)
        self.assertEqual(self.graph.get_layer(new_layer_id)['layer_type'], 'conv2d')
        self.assertIn(new_layer_id, self.graph.graph.successors(input_layer_id))

    def test_remove_layer_operator(self):
        """Test the RemoveLayerOperator."""
        layer_id = self.graph.add_layer('dense', {'units': 128})
        op = RemoveLayerOperator(layer_id)
        op.apply(self.graph)
        self.assertNotIn(layer_id, self.graph.graph.nodes)

    def test_modify_layer_params_operator(self):
        """Test the ModifyLayerParamsOperator."""
        layer_id = self.graph.add_layer('dense', {'units': 128})
        op = ModifyLayerParamsOperator(layer_id, {'units': 256})
        op.apply(self.graph)
        self.assertEqual(self.graph.get_layer(layer_id)['config']['units'], 256)

    def test_add_skip_connection_operator(self):
        """Test the AddSkipConnectionOperator."""
        layer1_id = self.graph.add_layer('conv2d', {})
        layer2_id = self.graph.add_layer('conv2d', {})
        op = AddSkipConnectionOperator(layer1_id, layer2_id)
        op.apply(self.graph)
        self.assertIn(layer2_id, self.graph.graph.successors(layer1_id))

    def test_change_layer_type_operator(self):
        """Test the ChangeLayerTypeOperator."""
        layer_id = self.graph.add_layer('conv2d', {})
        op = ChangeLayerTypeOperator(layer_id, 'maxpool')
        op.apply(self.graph)
        self.assertEqual(self.graph.get_layer(layer_id)['layer_type'], 'maxpool')

    def test_perturb_weights_operator(self):
        """Test the PerturbWeightsOperator."""
        layer_id = self.graph.add_layer('dense', {'units': 128})
        op = PerturbWeightsOperator(layer_id, 0.1)
        op.apply(self.graph)
        # For now, just check that it runs without error

    def test_reinitialize_layer_operator(self):
        """Test the ReinitializeLayerOperator."""
        layer_id = self.graph.add_layer('dense', {'units': 128})
        op = ReinitializeLayerOperator(layer_id)
        op.apply(self.graph)
        # For now, just check that it runs without error

    def test_prune_operator(self):
        """Test the PruneOperator."""
        op = PruneOperator(0.5, 'magnitude')
        op.apply(self.graph)
        # For now, just check that it runs without error

    def test_quantize_operator(self):
        """Test the QuantizeOperator."""
        op = QuantizeOperator(8, 'weights')
        op.apply(self.graph)
        # For now, just check that it runs without error

    def test_change_learning_rate_operator(self):
        """Test the ChangeLearningRateOperator."""
        op = ChangeLearningRateOperator(0.001)
        op.apply(self.graph)
        # For now, just check that it runs without error

    def test_modify_optimizer_params_operator(self):
        """Test the ModifyOptimizerParamsOperator."""
        op = ModifyOptimizerParamsOperator({'beta1': 0.9, 'beta2': 0.999})
        op.apply(self.graph)
        # For now, just check that it runs without error

    def test_mutate_loss_component_operator(self):
        """Test the MutateLossComponentOperator."""
        op = MutateLossComponentOperator({'weight': 0.5})
        op.apply(self.graph)
        # For now, just check that it runs without error


if __name__ == '__main__':
    unittest.main()
