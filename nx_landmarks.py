#Class to initialize a landmarks object
#
#
#
import random
import networkx as nx
import numpy as np


class selection_strategies:    
    def __init__(self):
        pass
    def degree_ranking(G, d)->list:
        degree = sorted([(v,(G.degree(v))) for v in G], key=lambda tup: tup[1], reverse=True)
        return degree[:d]
    def closeness_ranking(G, d)->list:
        closeness = nx.closeness_centrality(G)
        degree = sorted([(k,v) for k,v in zip(closeness.keys(), closeness.values())] , key= lambda tup: tup[1], reverse=True)
        return degree[:d]
    def random_ranking(G, d)->list:
        sample = random.sample(list(G.nodes),d)
        sample = [(node, G.degree(node)) for node in sample]
        return sample
    

class landmarks:
    def __init__(self, G, d = 1, selection_strategie = "rand"):
        if type(G) != nx.classes.graph.Graph:
            raise AttributeError(
                f"Expected graph to be of type networkx.classes.graph.Graph. Recieved {type(G)}"
            )
        self.graph = G
        if type(d) != int:
            raise AttributeError(
                f"Expected d to be of type int. Recieved {type(d)}"
            )
        self.d = d
        supported_rankings = {
            'rand': selection_strategies.random_ranking,
            'deg': selection_strategies.degree_ranking,
            'close': selection_strategies.closeness_ranking,
        }
        if selection_strategie not in supported_rankings.keys():
            raise AttributeError(
                f"Expected selection_strategie to be of type str. Recieved {type(selection_strategie)}"
            )
        self.selection_strategie = supported_rankings[selection_strategie]
        self.landmarks = self.selection_strategie(G,d)
        self.embeddings = self.__get_embeddings()

    def __get_embeddings(self):
        embeddings = np.zeros((self.graph.number_of_nodes(),len(self.landmarks)))
        for ranking, landmark in enumerate(self.landmarks):
            shortest_paths = nx.single_source_shortest_path_length(self.graph, source = landmark[0])
            for node, length in shortest_paths.items():
                embeddings[node-1,ranking] = length
        return embeddings
    
    def shortest_path_estimation_upper_bound(self, source , target):
        uppers = self.embeddings[source-1] + self.embeddings[target-1]
        return int(min(uppers))
        
    def shortest_path_estimation_lower_bound(self, source , target):
        lowers = self.embeddings[source-1] - self.embeddings[target-1]
        return int(max(abs(lowers)))
        