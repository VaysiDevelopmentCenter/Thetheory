from .base_operator import BaseOperator

class AddLayerOperator(BaseOperator):
    """Adds a new layer to the network graph."""
    def __init__(self, layer_type, config, input_layer_ids=None):
        self.layer_type = layer_type
        self.config = config
        self.input_layer_ids = input_layer_ids if input_layer_ids is not None else []

    def apply(self, network_graph):
        new_layer_id = network_graph.add_layer(self.layer_type, self.config)
        print(f"Added layer {new_layer_id} of type {self.layer_type}")
        for input_id in self.input_layer_ids:
            network_graph.add_connection(input_id, new_layer_id)
            print(f"Connected layer {input_id} to new layer {new_layer_id}")
        return new_layer_id

class RemoveLayerOperator(BaseOperator):
    """Removes a layer from the network graph."""
    def __init__(self, layer_id):
        self.layer_id = layer_id

    def apply(self, network_graph):
        network_graph.remove_layer(self.layer_id)
        print(f"Removed layer {self.layer_id}")

class ModifyLayerParamsOperator(BaseOperator):
    """Modifies the parameters of a layer."""
    def __init__(self, layer_id, param_changes):
        self.layer_id = layer_id
        self.param_changes = param_changes

    def apply(self, network_graph):
        layer = network_graph.get_layer(self.layer_id)
        layer['config'].update(self.param_changes)
        print(f"Modified parameters for layer {self.layer_id}")

class AddSkipConnectionOperator(BaseOperator):
    """Adds a skip connection between two layers."""
    def __init__(self, from_layer_id, to_layer_id):
        self.from_layer_id = from_layer_id
        self.to_layer_id = to_layer_id

    def apply(self, network_graph):
        network_graph.add_connection(self.from_layer_id, self.to_layer_id)
        print(f"Added skip connection from {self.from_layer_id} to {self.to_layer_id}")

class ChangeLayerTypeOperator(BaseOperator):
    """Changes the type of a layer."""
    def __init__(self, layer_id, new_type):
        self.layer_id = layer_id
        self.new_type = new_type

    def apply(self, network_graph):
        layer = network_graph.get_layer(self.layer_id)
        layer['layer_type'] = self.new_type
        print(f"Changed layer {self.layer_id} to type {self.new_type}")
