import random
import copy
import torch

# Global counter for innovation numbers
innovation_counter = 0

def get_new_innovation_number():
    global innovation_counter
    innovation_number = innovation_counter
    innovation_counter += 1
    return innovation_number

class ConnectionGene:
    def __init__(self, in_node, out_node, weight, enabled, innovation):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

class NodeGene:
    def __init__(self, node_id, node_type, activation_func):
        self.node_id = node_id
        self.node_type = node_type
        self.activation_func = activation_func

class NEATNetwork(torch.nn.Module):
    def __init__(self, node_genes, connection_genes):
        super(NEATNetwork, self).__init__()
        self.node_genes = node_genes
        self.connection_genes = connection_genes

    def forward(self, inputs):
        for node in self.node_genes:
            if node.node_type == 'input':
                node.output = inputs[node.node_id]
            else:
                node.output = node.activation_func(sum(connection.weight * self.node_genes[connection.in_node].output for connection in self.connection_genes if connection.out_node == node.node_id and connection.enabled))

        outputs = [node.output for node in self.node_genes if node.node_type == 'output']
        return outputs

    def mutate(self):
        if random.random() < 0.8:  # 80% chance to mutate weights
            for connection in self.connection_genes:
                if random.random() < 0.9:  # 90% chance to perturb weight
                    connection.weight += random.gauss(0, 1)
                else:  # 10% chance to assign new weight
                    connection.weight = random.uniform(-2, 2)

        if random.random() < 0.02:  # 2% chance to add new connection
            in_node = random.choice(self.node_genes)
            out_node = random.choice(self.node_genes)
            new_connection = ConnectionGene(in_node.node_id, out_node.node_id, weight=random.uniform(-2, 2), enabled=True, innovation=get_new_innovation_number())
            self.connection_genes.append(new_connection)

        if random.random() < 0.01:  # 1% chance to add new node
            old_connection = random.choice(self.connection_genes)
            old_connection.enabled = False
            new_node = NodeGene(node_id=len(self.node_genes), node_type='hidden', activation_func=torch.sigmoid)
            self.node_genes.append(new_node)
            new_connection1 = ConnectionGene(old_connection.in_node, new_node.node_id, weight=1.0, enabled=True, innovation=get_new_innovation_number())
            new_connection2 = ConnectionGene(new_node.node_id, old_connection.out_node, weight=old_connection.weight, enabled=True, innovation=get_new_innovation_number())
            self.connection_genes.append(new_connection1)
            self.connection_genes.append(new_connection2)

    def crossover(self, other):
        parent1, parent2 = (self, other) if len(self.node_genes) >= len(other.node_genes) else (other, self)

        child_node_genes = copy.deepcopy(parent1.node_genes)
        child_connection_genes = []

        for gene in parent1.connection_genes:
            if gene in parent2.connection_genes:
                child_connection_genes.append(copy.deepcopy(gene if random.random() < 0.5 else parent2.connection_genes[parent2.connection_genes.index(gene)]))
            else:
                child_connection_genes.append(copy.deepcopy(gene))

        return NEATNetwork(child_node_genes, child_connection_genes)

    def compatibility_distance(self, other, c1=1.0, c2=1.0, c3=0.4):
        matching_genes = len(set(self.connection_genes).intersection(other.connection_genes))
        disjoint_genes = len(self.connection_genes) + len(other.connection_genes) - 2 * matching_genes
        excess_genes = abs(len(self.connection_genes) - len(other.connection_genes))
        avg_weight_diff = sum(abs(self.connection_genes[i].weight - other.connection_genes[i].weight) for i in range(matching_genes)) / matching_genes

        return c1 * excess_genes / max(len(self.connection_genes), len(other.connection_genes)) + c2 * disjoint_genes / max(len(self.connection_genes), len(other.connection_genes)) + c3 * avg_weight_diff
