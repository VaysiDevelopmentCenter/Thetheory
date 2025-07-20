from .base_operator import BaseOperator

class ChangeLearningRateOperator(BaseOperator):
    """Changes the learning rate of the optimizer."""
    def __init__(self, new_lr):
        self.new_lr = new_lr

    def apply(self, network_graph):
        # This would modify the optimizer's configuration
        print(f"Changed learning rate to {self.new_lr}")

class ModifyOptimizerParamsOperator(BaseOperator):
    """Modifies parameters of the optimizer."""
    def __init__(self, param_changes):
        self.param_changes = param_changes

    def apply(self, network_graph):
        # This would modify the optimizer's configuration
        print(f"Modified optimizer parameters with: {self.param_changes}")

class MutateLossComponentOperator(BaseOperator):
    """(More advanced) Mutates a component of the loss function."""
    def __init__(self, loss_component_changes):
        self.loss_component_changes = loss_component_changes

    def apply(self, network_graph):
        # This would involve modifying the loss function, which could be very complex
        print(f"Mutated loss component with: {self.loss_component_changes}")
