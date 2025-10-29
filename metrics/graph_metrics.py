from netrd.distance import JaccardDistance, \
                            PortraitDivergence, \
                            IpsenMikhailov, \
                            NetLSD, \
                            LaplacianSpectral, \
                            DegreeDivergence, \
                            OnionDivergence
import networkx as nx

distance_list = [JaccardDistance(),
                PortraitDivergence(),
                IpsenMikhailov(),
                NetLSD(),
                LaplacianSpectral(),
                DegreeDivergence(),
                OnionDivergence()]


def calculate_graph_distances(graph_scores, graph_target):
    results = {}

    for distance in distance_list:
        distance_name = (str(distance)).split('.')[2] 
        results[distance_name] = distance.dist(graph_scores, graph_target)

    return results