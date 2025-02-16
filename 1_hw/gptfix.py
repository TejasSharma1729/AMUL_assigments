import json
import math
import itertools
import collections
import functools
import random
import heapq



########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################
def find_simplicial_vertex(adjacency_list, cliques):
    for vertex in range(len(adjacency_list)):
        if len(adjacency_list[vertex]) > 1:
            neighbors = adjacency_list[vertex]
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if neighbors[i] not in adjacency_list[neighbors[j]]:
                        return vertex, neighbors[i], neighbors[j]



class Inference:
    def __init__(self, data):
        """
        Initialize the Inference class with the input data.
        
        Parameters:
        -----------
        data : dict
            The input data containing the graphical model details, such as variables, cliques, potentials, and k value.
        
        What to do here:
        ----------------
        - Parse the input data and store necessary attributes (e.g., variables, cliques, potentials, k value).
        - Initialize any data structures required for triangulation, junction tree creation, and message passing.
        
        Refer to the sample test case for the structure of the input data.
        """
        self.num_variables = data["VariablesCount"]
        self.kval_in_top_k = data["k value (in top k)"]
        self.num_potentials = data["Potentials_count"]
        self.cliques = [(clique["cliques"], clique["potentials"]) for clique in data["Cliques and Potentials"]]
        self.adjacency_list = [[] for _ in range(self.num_variables)]

        for clique in self.cliques:
            for vertex in clique[0]:
                self.adjacency_list[vertex] = list(set(self.adjacency_list[vertex] + clique[0]))
                self.adjacency_list[vertex].remove(vertex)
        
        # self.cliques[0] --> the clique vertices, self.cliques[1] --> the potentials
        self.triangulated_graph = None
        self.maximal_cliques = None
        self.junction_tree = None
        self.jt_potentials = None
        self.Z_value = None
        self.marginals = None
        return

    def triangulate_and_get_cliques(self):
        """
        Triangulate the undirected graph and extract the maximal cliques.
        
        What to do here:
        ----------------
        - Implement the triangulation algorithm to make the graph chordal.
        - Extract the maximal cliques from the triangulated graph.
        - Store the cliques for later use in junction tree creation.

        Refer to the problem statement for details on triangulation and clique extraction.
        """
      
        max_cliques = set()
        nodes = set(range(self.num_variables))
        adjacency = {i: set(self.adjacency_list[i]) for i in range(self.num_variables)}

        while nodes:
            # Find node with minimum fill-in (min-degree heuristic as an approximation)
            min_fill_node = min(nodes, key=lambda n: len(adjacency[n]))

            # Identify its neighbors and create edges between them to form a clique
            neighbors = list(adjacency[min_fill_node])
            for u, v in itertools.combinations(neighbors, 2):
                adjacency[u].add(v)
                adjacency[v].add(u)

            # Create a maximal clique
            clique = tuple(sorted([min_fill_node] + neighbors))
            max_cliques.add(clique)

            # Remove the node
            nodes.remove(min_fill_node)
            for neighbor in adjacency[min_fill_node]:
                adjacency[neighbor].remove(min_fill_node)

            adjacency.pop(min_fill_node, None)  # Delete node from adjacency list

        self.maximal_cliques = list(max_cliques)
        return self.maximal_cliques
        
        
    def get_junction_tree(self):
        """
        Construct the junction tree from the maximal cliques.
        
        What to do here:
        ----------------
        - Create a junction tree using the maximal cliques obtained from the triangulated graph.
        - Ensure the junction tree satisfies the running intersection property.
        - Store the junction tree for later use in message passing.

        Refer to the problem statement for details on junction tree construction.
        """
        if not self.maximal_cliques:
            raise ValueError("Maximal cliques not computed. Run triangulation first.")

        clique_graph = []
        num_cliques = len(self.maximal_cliques)

        # Create edges based on separator sizes
        for i, j in itertools.combinations(range(num_cliques), 2):
            intersection = set(self.maximal_cliques[i]) & set(self.maximal_cliques[j])
            if intersection:
                clique_graph.append((len(intersection), i, j))  # (weight, clique1, clique2)

        # Sort edges by separator size (descending) for MWST
        clique_graph.sort(reverse=True, key=lambda x: x[0])

        # Kruskal’s algorithm for MWST
        parent = list(range(num_cliques))
        self.junction_tree = [[] for _ in range(num_cliques)]

        def find(v):
            while parent[v] != v:
                v = parent[v]
            return v

        def union(v1, v2):
            root1 = find(v1)
            root2 = find(v2)
            if root1 != root2:
                parent[root2] = root1

        for weight, i, j in clique_graph:
            if find(i) != find(j):
                union(i, j)
                self.junction_tree[i].append(j)
                self.junction_tree[j].append(i)

        return self.junction_tree


    def assign_potentials_to_cliques(self):
        """
        Assign potentials to the cliques in the junction tree.
        
        What to do here:
        ----------------
        - Map the given potentials (from the input data) to the corresponding cliques in the junction tree.
        - Ensure the potentials are correctly associated with the cliques for message passing.
        
        Refer to the sample test case for how potentials are associated with cliques.
        """
        self.jt_potentials = []
        for maximal_clique in self.maximal_cliques:
            relevant_potentials = []
            for clique, potential in self.cliques:
                if set(clique).issubset(set(maximal_clique)):  # Subset check
                    relevant_potentials.append((clique, potential))

            # Combine all relevant potentials
            clique_size = len(maximal_clique)
            full_potential = [1] * (2 ** clique_size)

            for idx in range(2 ** clique_size):
                for clique, potential in relevant_potentials:
                    clique_idx = 0
                    for i, v in enumerate(clique):
                        clique_idx = (clique_idx << 1) | ((idx >> v) & 1)
                    full_potential[idx] *= potential[clique_idx]

            self.jt_potentials.append((maximal_clique, full_potential))

        return self.jt_potentials
    

    def get_z_value(self):
        """
        Compute the partition function (Z value) of the graphical model.
        
        What to do here:
        ----------------
        - Implement the message passing algorithm to compute the partition function (Z value).
        - The Z value is the normalization constant for the probability distribution.
        
        Refer to the problem statement for details on computing the partition function.
        """
        messages = {}  # Store messages
        for clique, potential in self.jt_potentials:
            messages[tuple(clique)] = sum(potential)

        self.Z_value = sum(messages.values())
        return self.Z_value
    

    def compute_marginals(self):
        """
        Compute the marginal probabilities for all variables in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to compute the marginal probabilities for each variable.
        - Return the marginals as a list of lists, where each inner list contains the probabilities for a variable.
        
        Refer to the sample test case for the expected format of the marginals.
        """
        self.marginals = [[0, 0] for _ in range(self.num_variables)]
        
        for var in range(self.num_variables):
            marginal_sum = [0, 0]
            for clique, potential in self.jt_potentials:
                if var in clique:
                    for idx, val in enumerate(potential):
                        state = (idx >> var) & 1  # Get bit for var
                        marginal_sum[state] += val
            self.marginals[var][0] = marginal_sum[0] / self.Z_value
            self.marginals[var][1] = marginal_sum[1] / self.Z_value

        return self.marginals


    def compute_top_k(self):
        """
        Compute the top-k most probable assignments in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to find the top-k assignments with the highest probabilities.
        - Return the assignments along with their probabilities in the specified format.
        
        Refer to the sample test case for the expected format of the top-k assignments.
        """
        pass



########################################################################

# Do not change anything below this line

########################################################################

class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
    
    def get_output(self):
        n = len(self.data)
        output = []
        for i in range(n):
            inference = Inference(self.data[i]['Input'])
            inference.triangulate_and_get_cliques()
            inference.get_junction_tree()
            inference.assign_potentials_to_cliques()
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
            top_k_assignments = inference.compute_top_k()
            output.append({
                'Marginals': marginals,
                'Top_k_assignments': top_k_assignments,
                'Z_value' : z_value
            })
        self.output = output

    def write_output(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.output, file, indent=4)


if __name__ == '__main__':
    evaluator = Get_Input_and_Check_Output('Sample_Testcase.json')
    evaluator.get_output()
    evaluator.write_output('Sample_Testcase_Output.json')
