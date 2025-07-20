from modules.engine.network_graph import NetworkGraph
from modules.evaluation.evaluation_oracle import EvaluationOracle
from modules.evaluation.reward import Reward

def main():
    """
    A simple script to demonstrate the evaluation and reward process.
    """
    # 1. Create a NetworkGraph
    graph = NetworkGraph()
    input_layer = graph.add_layer('input', {})
    hidden_layer = graph.add_layer('dense', {'units': 128})
    output_layer = graph.add_layer('dense', {'units': 10})
    graph.add_connection(input_layer, hidden_layer)
    graph.add_connection(hidden_layer, output_layer)

    # 2. Evaluate the NetworkGraph
    oracle = EvaluationOracle('cifar10', 'classification')
    metrics = oracle.evaluate(graph)

    # 3. Calculate the reward
    weights = {'accuracy': 1.0, 'size': 0.001}
    reward_calculator = Reward(weights)
    reward = reward_calculator.calculate(metrics)

    # 4. Print the results
    print("\n--- Evaluation Results ---")
    print(f"Metrics: {metrics}")
    print(f"Reward: {reward}")
    print("------------------------")

if __name__ == '__main__':
    main()
