#Class to initialize a landmarks object
#
#
#
import networkx as nx
import numpy as np


class selection_strategies:    
    def __init__(self):
        pass
    def degree_ranking(G)->list:
        degree = sorted([(v,(G.degree(v))) for v in G], key=lambda tup: tup[1], reverse=True)
        return [d[0] for d in degree]
    def closeness_ranking(G)->list:
        closeness = nx.closeness_centrality(G)
        closeness = sorted([(k,v) for k,v in zip(closeness.keys(), closeness.values())] , key= lambda tup: tup[1], reverse=True)
        return [c[0] for c in closeness]
    def random_ranking(G)->list:
        sample = np.array(G.nodes)
        np.random.shuffle(sample)
        return sample
    

class landmarks:
    def __init__(self, G, d = 1, selection_strategie = "rand",h = 0):
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
        if type(h) != int:
            raise AttributeError(
                f"Expected h to be of type int. Recieved {type(h)}"
            )
        self.h = h
        
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
        self.landmark_ranking = self.selection_strategie(G)
        self.landmarks = None
        self.embeddings = None

    def get_landmarks(self):
        landmarks = []
        embeddings = np.full((self.graph.number_of_nodes(),self.d), np.inf)
        x = 0 
        for i in range(self.d):
            while min(embeddings[self.landmark_ranking[x]]) < self.h:
                x+=1
            landmarks.append(self.landmark_ranking[x])    
            shortest_paths = nx.single_source_shortest_path_length(self.graph, source = landmarks[-1])
            for node, length in shortest_paths.items():
                embeddings[node,i] = length
            
        self.landmarks = np.array(landmarks)
        self.embeddings = embeddings
    
    def shortest_path_estimation_upper_bound(self, source , target):
        uppers = self.embeddings[source] + self.embeddings[target]
        return int(min(uppers))
        
    def shortest_path_estimation_lower_bound(self, source , target):
        lowers = self.embeddings[source] - self.embeddings[target]
        return int(max(abs(lowers)))
    
    def shortest_path_estimation_capture_method(self,source, target):
        lower =  max(abs(self.embeddings[source]-self.embeddings[target]))
        upper =  min(self.embeddings[source]+self.embeddings[target])
        if upper == lower: return upper
        #Find capture upper bound:
        d_bound_lower = np.where(abs(self.embeddings[source]-self.embeddings[source]) == lower) 
        d_bound_upper = np.where((self.embeddings[source]+self.embeddings[source]) == upper)
        for bound in [d_bound_lower,d_bound_upper]:
            pass
            #check weather the captures are found and then confirm / deny lower upper bound 



    def add_landmarks(self, n = 1):
        x = np.where(self.landmark_ranking == self.landmarks[-1])
        x = int(x[0])
        self.embeddings = np.append(self.embeddings, np.full((self.graph.number_of_nodes(),n), np.inf), axis=1)
        for i in range(self.embeddings.shape[1]-n, self.embeddings.shape[1]):
            while min(self.embeddings[self.landmark_ranking[x]]) < self.h:
                x+=1
            self.landmarks = np.append(self.landmarks, self.landmark_ranking[x])
            shortest_paths = nx.single_source_shortest_path_length(self.graph, source = self.landmarks[-1])
            for node, length in shortest_paths.items():
                self.embeddings[node,i] = length