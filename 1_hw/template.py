#!/usr/bin/env python3
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

def find_simplicial_vertices(adjacency_list):   #function to find simplicial vertices (brute force)
    simplicial_vertices = []
    for (vertex, neighbors) in adjacency_list.items():
        flag = True
        if len(neighbors) > 1:      #check if vertex is isolated
            for i in range(len(neighbors)):
                if not flag:
                    break
                for j in range(i+1, len(neighbors)):
                    if neighbors[i] not in adjacency_list[neighbors[j]]:    #check if neighbors are not connected
                        flag = False
                        break
        if flag:
            simplicial_vertices.append(vertex)  #add simplicial vertices to the list
    return simplicial_vertices



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
        #initialize the variables from the input data
        self.num_variables = data["VariablesCount"]
        self.kval_in_top_k = data["k value (in top k)"]
        self.num_potentials = data["Potentials_count"]
        self.cliques = [(clique["cliques"], clique["potentials"]) for clique in data["Cliques and Potentials"]]
        self.adjacency_list = [[] for _ in range(self.num_variables)]

        #initialize the adjacency list
        for clique in self.cliques:
            for vertex in clique[0]:
                self.adjacency_list[vertex] = list(set(self.adjacency_list[vertex] + clique[0]))
                self.adjacency_list[vertex].remove(vertex)
        
        # self.cliques[0] --> the clique vertices, self.cliques[1] --> the potentials
        self.triangulated_graph = None
        self.optimal_ordering = None
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
        nodes = set(range(self.num_variables))
        self.optimal_ordering = []
        # adj_list = CLONE of graph, which is MODIFIED, should be EMPTY at the end
        adj_list = {i: list(set(neighbors)) for i, neighbors in enumerate(self.adjacency_list)}
        self.maximal_cliques = []

        while adj_list != {}:
            for vertex in list(adj_list.keys()):
                if len(adj_list[vertex]) == 0:  
                    #if a vertex is isolated, add it to the ordering
                    self.optimal_ordering.append(vertex)
                    adj_list.pop(vertex)        
                    #remove the vertex from the adjacency list

            simplicial_vertices = find_simplicial_vertices(adj_list)
            if len(simplicial_vertices) > 0:
                self.optimal_ordering.extend(simplicial_vertices)   
                #append the simplicial vertices to the ordering
                for vertex in simplicial_vertices:      
                    #remove all occurences of simplicial vertices from the adjacency list
                    for neighbor in adj_list[vertex]:
                        adj_list[neighbor].remove(vertex)
                    self.maximal_cliques.append(set([vertex] + adj_list[vertex]))   
                    #add the simplicial vertex and its neighbors to the maximal cliques
                    adj_list.pop(vertex)

            else:   
                #find min degree vertex
                min_degree_vertex = -1
                for (vertex, neighbors) in adj_list.items():
                    if len(neighbors) > 0:
                        if min_degree_vertex == -1 or len(adj_list[vertex]) < len(adj_list[min_degree_vertex]):
                            min_degree_vertex = vertex
                if min_degree_vertex == -1:
                    break

                neighbors = adj_list[min_degree_vertex]
                adj_list.pop(min_degree_vertex)
                for neighbor in neighbors:
                    adj_list[neighbor].remove(min_degree_vertex)
                self.optimal_ordering.append(min_degree_vertex)  
                #add the min degree vertex to the ordering
                self.maximal_cliques.append(set([min_degree_vertex] + neighbors)) 
                #add the min degree vertex and its neighbors to the maximal cliques

                for i in range(len(neighbors)): 
                    #triangulate the graph by adding edges between the neighbors of the min degree vertex
                    for j in range(i+1, len(neighbors)):
                        if neighbors[j] not in adj_list[neighbors[i]]:
                            adj_list[neighbors[i]].append(neighbors[j])
                            adj_list[neighbors[j]].append(neighbors[i])
                            self.adjacency_list[neighbors[i]].append(neighbors[j])
                            self.adjacency_list[neighbors[j]].append(neighbors[i])
                            
        assert len(self.optimal_ordering) == self.num_variables
        for clique in self.maximal_cliques: #remove redundant maximal cliques
            for other_clique in self.maximal_cliques:
                if clique != other_clique and clique.issubset(other_clique):
                    self.maximal_cliques.remove(clique)
                    break
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
        for i in range(len(self.maximal_cliques)):
            node = self.maximal_cliques[i]
            for j in range(i+1, len(self.maximal_cliques)):
                other_node = self.maximal_cliques[j]
                if node != other_node and len(node.intersection(other_node)) > 0:
                    if node not in self.junction_tree:
                        self.junction_tree[i] = []
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
        for clique in self.maximal_cliques:
            cl_list = list(clique)
            cl_list.sort()
            self.jt_potentials.append((clique, [1] * 2 ** len(clique)))

            # Created the masterlist for potential. Now, we need to assign values to it.
            # Just like normal potential, express as len(clique) bits.
            # MSB ... LSB represent var 0, var 1, ... var (last) in clique, in order
            for node in clique:
                for c, p in self.cliques:
                    if not set(c).issubset(clique):
                        continue

                    self.cliques.remove((c, p))
                    cinds = [cl_list.index(i) for i in c]
                    for j in range(2 ** len(clique)):
                        # Find the corresponding assignment of all variables
                        # Reverse because MSB (not LSB) is the first variable
                        parity_ofj = [(j >> k) & 1 for k in range(len(clique) - 1, -1, -1)]
                        index = 0

                        for k in range(len(c)):
                            if (parity_ofj[cl_list.index(c[k])] == 1):
                                # Corresponding bit in subset clique "c" is 1
                                # Once again, reverse because MSB is the first variable
                                index += 2 ** (len(c) - 1 - k)
                        self.jt_potentials[-1][1][j] *= p[index]
        
        assert len(self.cliques) == 0
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
        all_factors = set([(tuple(a), tuple(b)) for (a,b) in self.jt_potentials])
        # Creates a deep copy. all_factors is modified, but jt_potentials is not.

        for i in self.optimal_ordering:
            factors = []
            variables = set()
            for factor in all_factors:
                if i in factor[0]:
                    factors.append(factor)
                    variables = variables.union(set(factor[0]))
            # Now, we have all factors that contain variable i

            # We need to multiply them all together
            product = [1] * 2 ** (len(variables))
            var_list = list(variables)
            idx_ofi = list(variables).index(i)

            for j in range(2 ** (len(variables))):
                # Once again, reverse because MSB is the first variable
                parity_ofj = [(j >> k) & 1 for k in range(len(variables) - 1, -1, -1)]
                for factor in factors:
                    factor_index = 0
                    for k in range(len(factor[0])):
                        if (parity_ofj[var_list.index(factor[0][k])] == 1):
                            factor_index += 2 ** (len(factor[0]) - 1 - k)
                    product[j] *= factor[1][factor_index]

                # This is a modification after multiplication to ease marginal computation
                # I assign (assignment but i = 0) ka value += (assignment) ka value
                # whenever "i" is assigned 1. Then make assignment ka value 0.
                if (j >> (len(var_list) - 1 - idx_ofi)) & 1 == 1:
                    product[j ^ (1 << (len(var_list) - 1 - idx_ofi))] += product[j]
                    product[j] = 0
            
            # Remove the product where i = 1
            # Correctness: all remaining bits occur in same order as expected.
            product_wo_i = [j for j in product if j != 0]
            variables.remove(i)
            var_list = list(variables) + [i]
            all_factors = all_factors - set(factors)
            all_factors.add((tuple(variables), tuple(product_wo_i)))
        
        # Last loop is in case more than one factor persists
        # But since all variables are marginalized out, factors should be empty
        # Potential should contain only one value, obviously
        self.Z_value = 1
        for factor in all_factors:
            assert len(factor[0]) == 0
            self.Z_value *= factor[1][0]
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
        self.marginals = [[1, 1] for _ in range(self.num_variables)]
        for i in range (self.num_variables):
            all_factors = set([(tuple(a), tuple(b)) for (a,b) in self.jt_potentials])
            # Once again, deep copy [once for each "i" -- whole thing repeats for each "i"]

            for j in self.optimal_ordering:
                if j == i:
                    continue
                    # We DO NOT want to marginalize out the variable we are interested in
                    # However, we marginalize the rest, just like in Z computation
                
                factors = []
                variables = set()
                for factor in all_factors:
                    if j in factor[0]:
                        factors.append(factor)
                        variables = variables.union(set(factor[0]))
                
                # Once again, compute the product with "j" marginalized out
                product = [1] * 2 ** (len(variables))
                var_list = list(variables)
                idx_ofj = list(variables).index(j)

                for k in range(2 ** (len(variables))):
                    # Once again, reverse because MSB is the first variable
                    parity_ofk = [(k >> l) & 1 for l in range(len(variables) - 1, -1, -1)]
                    for factor in factors:
                        factor_index = 0
                        for l in range(len(factor[0])):
                            if (parity_ofk[var_list.index(factor[0][l])] == 1):
                                factor_index += 2 ** (len(factor[0]) - 1 - l)
                        product[k] *= factor[1][factor_index]
                    
                    # Once again, this is a modification after multiplication to ease marginal computation
                    if (k >> (len(var_list) - 1 - idx_ofj)) & 1 == 1:
                        product[k ^ (1 << (len(var_list) - 1 - idx_ofj))] += product[k]
                        product[k] = 0
                
                # Remove the product where j = 1
                product_wo_j = [k for k in product if k != 0]
                variables.remove(j)
                all_factors = all_factors - set(factors)
                all_factors.add((tuple(variables), tuple(product_wo_j)))
            
            # Now, we have the product of all factors that do not contain "i"
            # All remaining factors have only "i" as a variable
            # Product of those potentials (factors) = required marginal
            for factor in all_factors:
                assert len(factor[0]) == 1 and factor[0][0] == i
                self.marginals[i][0] *= factor[1][0]
                self.marginals[i][1] *= factor[1][1]
            
            # Normalize the marginals
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
    evaluator = Get_Input_and_Check_Output('TestCases.json')
    evaluator.get_output()
    evaluator.write_output('TestCases_Output.json')
