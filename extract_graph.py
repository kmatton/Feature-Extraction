import networkx as nx
import numpy as np
import truecase
import preprocess_text as pt
from process_text import *


"""
Contains functions to compute word graph measures.
"""


def create_naive_graph(segments):
    """
    creates a naive graph representation from the set of utterances of a single speaker
    :param segments: a list of segments, where each segment is a list of the words that make up that segment
    :return: g: a graph made up of nodes representing each distinct word used throughout all the segments
    and edges that link consecutive words (words are only considered consecutive if they are adjacent within the
    same segment). The graph is in the form of a directed multigraph (edges have direction and the graph can have
    self-loops and parallel edges).
    """
    g = nx.MultiDiGraph()
    for segment in segments:
        for i in range(len(segment) - 1):
            g.add_edge(segment[i], segment[i + 1])
        if len(segment) == 1:
            g.add_node(segment[0])
    return g


def create_lemma_graph(segments):
    """
    creates a lemma graph representation from the set of utterances of a single speaker
    :param segments: a list of segments, where each segment is a list of the words that make up that segment
    :return: g: a graph made up of nodes representing each distinct lemma word used throughout all the segements and
    edges that link consecutive words (in the form of their lemmas). Again, words are only considered consecutive if
    they are adjacent within the same segment.
    """
    segments = lemmatize(segments)
    g = create_naive_graph(segments)
    return g


def create_pos_graph(segments):
    """
    creates a part of speech graph representation from the set of utterances of a single speaker
    :param segments: a list of segments, where each segment is a list of the words that make up that segment
    :return: g: a graph made up of nodes representing each distinct part of speech used throughout all the segments and
    edges that link parts of speech that are used in consecutively (adjacent within the same sentence)
    """
    for i in range(len(segments)):
        # transform words to their associated parts of speech
        # first transform to true case (i.e. capitalize if supposed to be), so that POS tagger works better
        segment_str = " ".join(segments[i])
        truecase_str = truecase.get_true_case(segment_str)
        segments[i] = nltk.pos_tag(truecase_str.split(" "))
        for j in range(len(segments[i])):
            segments[i][j] = segments[i][j][1]
    g = create_naive_graph(segments)
    return g


def get_connectivity_measures(graph, u_graph, graph_type, feats_dict):
    """
    Computes graph measures related to connectivity: average node degree (ATD), largest connected component (LCC),
    and largest strongly connected component (LSC).
    :param graph: a directed multigraph (i.e. can have self loops and parallel edges)
    :param u_graph: an undirected multigraph
    :param graph_type: naive, lemma, or POS
    :param feats_dict: dictionary to store feature values for the transcript
    """
    # calculate average degree of every node in the graph (ATD)
    atd = 0
    node_list = graph.nodes()
    for node in node_list:
        degree = graph.degree(node)
        atd += degree
    if len(graph):
        atd /= len(graph)
    else:
        atd = float('nan')
    feats_dict['ave_degree_{}'.format(graph_type)] = atd

    # calculate number of nodes in maximum connected component(LCC)
    components = sorted(nx.connected_components(u_graph), key=len, reverse=True)
    if components:
        feats_dict['lcc_{}'.format(graph_type)] = len(components[0])
    else:
        feats_dict['lcc_{}'.format(graph_type)] = 0

    # calculate number of nodes in maximum strongly connected component (LSC)
    s_components = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
    if s_components:
        feats_dict['lsc_{}'.format(graph_type)] = len(s_components[0])
    else:
        feats_dict['lsc_{}'.format(graph_type)] = 0


def get_parallel_edges(graph, graph_type, feats_dict):
    """
    Calculate number of parallel edges in graph. Edges must be in same direction to count as parallel
    (L2 measure counts loops with two nodes). Each repeated edge count as one parallel edge.
    Store parallel edge count in feats_dict.
    In density metric, compute E' = E - (L1 + PE). However, self-loops (L1) can be parallel edges, and they shouldn't
    be double counted in E' measure. Therefore, also returns count of edges that are self-loops and parallel.
    :param graph: a directed multigraph (i.e. can have self loops and parallel edges)
    :param graph_type: naive, lemma, or POS
    :param feats_dict: dictionary to store feature values for the transcript
    :return: num_p_edges: total count of parallel edges,
    pe_l1_count: count of edges that are both parallel and self-loops
    """
    num_p_edges = 0
    edge_list = list(graph.edges())
    edge_set = set(edge_list)
    pe_l1_count = 0
    for edge in edge_set:
        occurrences = edge_list.count(edge)
        if occurrences > 1:
            if edge[0] == edge[1]:
                pe_l1_count += (occurrences - 1)
            num_p_edges += (occurrences - 1)
    feats_dict['num_p_edges_{}'.format(graph_type)] = num_p_edges
    return num_p_edges, pe_l1_count


def get_loops(graph, graph_type, feats_dict):
    """
    Calculate number of loops with two nodes (L2) and with three nodes (L3).
    :param graph: a directed multigraph (i.e. can have self loops and parallel edges)
    :param graph_type: naive, lemma, or POS
    :param feats_dict: dictionary to store feature values for the transcript
    """
    adj_mat = nx.to_numpy_matrix(graph)
    # make sure self loops aren't counted
    # otherwise traversing a self-loop two/three times in a row will be counted as a loop with two/three nodes
    np.fill_diagonal(adj_mat, 0)
    squared_mat = np.matmul(adj_mat, adj_mat)
    # divide by two/three because loop is counted once for each node in loop in sum of trace
    feats_dict['l2_{}'.format(graph_type)] = np.trace(squared_mat) / 2
    cubed_mat = np.matmul(adj_mat, squared_mat)
    feats_dict['l3_{}'.format(graph_type)] = np.trace(cubed_mat) / 3


def get_shortest_path_metrics(u_graph, graph_type, feats_dict):
    """
    Compute graph measures related to shortest path lengths: diameter (DI: longest shortest path between any two nodes),
    average shortest path (ASP: average length of the shortest path between pairs of nodes in a graph, computed across
    all connected components)
    Measures are computed using undirected graph following paper by Mota et al.
    :param u_graph: an undirected multigraph (loops and parallel edges don't affect computation as the shortest
    path won't include them)
    :param graph_type: naive, lemma, or POS
    :param feats_dict: dictionary to store feature values for the transcript
    """
    longest = 0
    average = 0
    num_pairs = 0
    for component in nx.connected_component_subgraphs(u_graph):
        lengths = dict(nx.all_pairs_shortest_path_length(component))
        nodes = list(component.nodes())
        num_nodes = len(nodes)
        num_pairs += float((num_nodes * (num_nodes - 1) / 2))
        for i in range(num_nodes):
            # don't check for shortest path between node and itself
            for j in range(i + 1, num_nodes):
                path_length = lengths[nodes[i]][nodes[j]]
                if path_length > longest:
                    longest = path_length
                average += path_length
    if num_pairs:
        average /= float(num_pairs)
    # diameter will be zero if graph is empty or largest connected component is of size 1
    feats_dict['di_{}'.format(graph_type)] = longest
    # calculate average shortest path (ASP)
    feats_dict['asp_{}'.format(graph_type)] = average


def get_graph_metrics(graph, graph_type, feats_dict):
    """
    Computes features for the input graph. Features include: num_nodes: the number of nodes present in the graph,
    num_edges: number of edges in the graph, num_p_edges: number of parallel edges present in the graph,
    lcc: number of nodes in the largest connected component of the graph,
    lsc: number of nodes in the largest strongly connected component of the graph,
    atd: average degree of the nodes in the graph, l1: number of self-loops, l2: number of loops with two nodes,
    l3: number of triangles (an approximation to the number of loops with three nodes), and graph density.
    Graph connectivity measures(ASP and diameter) are also computed.
    :param graph: a directed multigraph
    :param graph_type: naive, lemma, or POS
    :param feats_dict: dictionary to store feature values for the transcript
    """
    # calculate number of nodes
    num_nodes = len(graph)
    feats_dict['num_nodes_{}'.format(graph_type)] = num_nodes
    # calculate number of edges
    feats_dict['num_edges_{}'.format(graph_type)] = graph.number_of_edges()
    # get undirected graph
    u_graph = graph.to_undirected()
    get_connectivity_measures(graph, u_graph, graph_type, feats_dict)
    num_p_edges, pe_l1_count = get_parallel_edges(graph, graph_type, feats_dict)
    # calculate number of self-loops (L1)
    l_one = len(list(graph.selfloop_edges()))
    feats_dict['l1_{}'.format(graph_type)] = l_one
    get_loops(graph, graph_type, feats_dict)
    # calculate graph density (D)
    # This measure is defined for simple graphs. Therefore, we take E' = E - (L1 + PE).
    # i.e. duplicate edges in same direction only count once and self-loops are not counted
    e_prime = graph.number_of_edges() - (l_one + num_p_edges - pe_l1_count)
    if e_prime < 0:
        feats_dict['d_{}'.format(graph_type)] = float('nan')
    elif num_nodes:
        feats_dict['d_{}'.format(graph_type)] = e_prime / float(num_nodes * num_nodes)
    else:
        feats_dict['d_{}'.format(graph_type)] = float('nan')
    get_shortest_path_metrics(u_graph, graph_type, feats_dict)


def get_word_count(segments):
    count = 0
    for segment in segments:
        count += len(segment)
    return count


def add_norm_feats(feats_dict, word_count):
    """
    :param feats_dict: Dictionary mapping feature name to value for transcript
    :param word_count: Transcript word count
    """
    for feat, value in list(feats_dict.items()):
        feats_dict["{}_norm".format(feat)] = (float(value) / float(word_count)) if word_count else float('nan')


def get_stopwords():
    with open("graph_stops.txt", 'r') as sf:
        return set([line.strip() for line in sf])


def remove_stopwords(segments):
    """
    Removes stopwords from each segment and removes segments that contain only stopwords.
    :param segments: list of segments, where each segment is a list of words
    :return: cleaned_segments: list of segments with stopwords removed
    """
    cleaned_segments = []
    stops = get_stopwords()
    for segment in segments:
        c_seg = [word for word in segment if word not in stops]
        if c_seg:
            cleaned_segments.append(c_seg)
    return cleaned_segments


def extract_graph_feats(transcript, remove_stops=False):
    """
    :param segments:  List of transcript segments. Each segment is represented as a tuple of the form (start time,
    stop time, text).
    :param remove_stops: Whether or not to remove stopwords from transcript segments before extracting features.
    :return: feats_dict: Dictionary mapping feature name to value for transcript
    """
    # first pre-process text
    # remove can ignore timing info because not relevant to computing graph feats
    transcript = [seg[2] for seg in transcript]
    # split into words
    segments = [seg.strip().split(" ") for seg in transcript]
    # remove non-verbal expressions
    segments = pt.remove_non_verbal_exp(segments)
    feats_dict = {}
    if remove_stops:
        segments = remove_stopwords(segments)
    # build graphs
    naive_graph = create_naive_graph(segments)
    lemma_graph = create_lemma_graph(segments)
    pos_graph = create_pos_graph(segments)
    # compute features for each graph
    get_graph_metrics(naive_graph, 'naive', feats_dict)
    get_graph_metrics(lemma_graph, 'lemma', feats_dict)
    get_graph_metrics(pos_graph, 'pos', feats_dict)
    # add normalized versions of features
    word_count = get_word_count(segments)
    add_norm_feats(feats_dict, word_count)
    return feats_dict
