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
        
        while nodes:
            min_fill_node = None
            min_degree = float('inf')
            for n in nodes:
                degree = len(self.adjacency_list[n])
                if degree < min_degree:
                    min_degree = degree
                    min_fill_node = n

            neighbors = list(self.adjacency_list[min_fill_node])
            for u, v in itertools.combinations(neighbors, 2):
                if v not in self.adjacency_list[u]:
                    self.adjacency_list[u].append(v)
                    self.adjacency_list[v].append(u)
            
            clique = tuple(sorted([min_fill_node] + neighbors))
            max_cliques.add(clique)
            nodes.remove(min_fill_node)
            self.adjacency_list[min_fill_node] = []
        
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
        self.junction_tree = [[] for _ in range(len(self.maximal_cliques))]
        junction_edges = []
        for i in range(len(self.maximal_cliques)):
            for j in range(i+1, len(self.maximal_cliques)):
                intersection = set(self.maximal_cliques[i]) & set(self.maximal_cliques[j])
                if len(intersection) > 0:
                    junction_edges.append((i, j, len(intersection)))
        
        # Kruskal's algorithm
        sorted_edges = sorted(junction_edges, key = lambda x: x[2])
        parent = [i for i in range(len(self.maximal_cliques))]
        rank = [0] * len(self.maximal_cliques)
        for i, j, _ in sorted_edges:
            k, l = i, j
            while parent[k] != k:
                k = parent[k]
            while parent[l] != l:
                l = parent[l]
            if k == l:
                parent[i] = l
                parent[j] = l
                continue
            if rank[k] < rank[l]:
                parent[i] = l
                parent[j] = l
                parent[k] = l
            elif rank[k] > rank[l]:
                parent[i] = k
                parent[j] = k
                parent[l] = k
            self.junction_tree[i].append(j)
            self.junction_tree[j].append(i)
        
        # TODO: Verify RIP property

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
            appropriate_cliques = []
            for clique, potential in self.cliques:
                if set(clique).issubset(set(maximal_clique)):
                    appropriate_cliques.append((clique, potential))
            maximal_potential = [1] * 2 ** len(maximal_clique)
            for i in range(2 ** len(maximal_clique)):
                for clique, potential in appropriate_cliques:
                    clique_index = 0
                    for j in range(len(clique)):
                        clique_index *= 2
                        clique_index += ((i >> clique[j]) & 1)
                    maximal_potential[i] *= potential[clique_index]
            self.jt_potentials.append((maximal_clique, maximal_potential))
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
        ordering = list(range(self.num_variables)) # TODO: Implement ordering
        all_factors = set(self.jt_potentials)
        for i in ordering:
            factors = []
            variables = set()
            for factor in all_factors:
                if i in factor[0]:
                    factors.append(factor)
                    variables = variables.union(set(factor[0]))
            variables.remove(i)
            product_wo_i = [1] * 2 ** (len(variables))
            for j in range(2 ** (len(variables))):
                for factor in factors:
                    factor_index = 0
                    for k in range(len(factor[0])):
                        if factor[0][k] != i:
                            factor_index *= 2
                            factor_index += ((j >> factor[0][k]) & 1)
                    product_wo_i[j] *= factor[1][factor_index]
            all_factors = all_factors - set(factors)
            all_factors.add((tuple(variables), product_wo_i))
        self.Z_value = sum(list(all_factors)[0][1])
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
        ordering = list(range(self.num_variables)) # TODO: Implement ordering
        self.marginals = [[1, 1] for _ in range(self.num_variables)]
        for i in range (len(self.num_variables)):
            all_factors = set(self.jt_potentials)
            for j in range(self.num_variables):
                if j == i:
                    continue
                factors = []
                variables = set()
                for factor in all_factors:
                    if j in factor[0]:
                        factors.append(factor)
                        variables = variables.union(set(factor[0]))
                variables.remove(j)
                product_wo_j = [1] * 2 ** (len(variables))
                for k in range(2 ** (len(variables))):
                    for factor in factors:
                        factor_index = 0
                        for l in range(len(factor[0])):
                            if factor[0][l] != j:
                                factor_index *= 2
                                factor_index += ((k >> factor[0][l]) & 1)
                        product_wo_j[k] *= factor[1][factor_index]
                all_factors = all_factors - set(factors)
                all_factors.add((tuple(variables), product_wo_j))
            self.marginals[i] = [sum(list(all_factors)[0][1][::2]), sum(list(all_factors)[0][1][1::2])]
        for i in range(self.num_variables):
            self.marginals[i][0] /= self.Z_value
            self.marginals[i][1] /= self.Z_value
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
