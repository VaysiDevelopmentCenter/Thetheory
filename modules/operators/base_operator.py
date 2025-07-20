from abc import ABC, abstractmethod

class BaseOperator(ABC):
    """
    Abstract base class for all mutation operators.
    """
    @abstractmethod
    def apply(self, network_graph):
        """
        Applies the mutation to the given NetworkGraph.

        :param network_graph: The NetworkGraph to mutate.
        """
        pass
