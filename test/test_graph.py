import jraph

from magpi.graph import nearest_neighbors, grid_neighbors, make_grid_graph, make_neighbors_graph, GNNLayer, GNNIntegrationLayer

from . import *


class TestNearestNeighbors(JaxTestCase):
    def test_000_nearest_neighbors(self):
        X = array([1,2,3])
        neighbors, padding = nearest_neighbors(X, 1)
        true_neighbors = array([[1,0,1]]).T
        true_padding = array([[True, True, True]]).T
        self.assertIsclose(neighbors, true_neighbors)
        self.assertPytreeEqual(padding, true_padding)

    def test_001_nearest_neighbors_with_radius(self):
        X = array([1,2,3.1])
        neighbors, padding = nearest_neighbors(X, 1, radius=1)
        true_neighbors = array([[1,0,-1]]).T
        true_padding = array([[True, True, False]]).T
        self.assertIsclose(neighbors, true_neighbors)
        self.assertPytreeEqual(padding, true_padding)


class TestMakeGraph(JaxTestCase):
    def test_000_make_graph(self):
        nodes = array([0, 1, 2])
        graph = make_neighbors_graph(nodes, 2)
        self.assertPytreeEqual(graph.nodes, array([[0, 1, 2, 0]]).T)  # last node is a padding node
        self.assertPytreeEqual(graph.edges, array([[2, -2, 1, -1, 1, -1]]).T)
        self.assertPytreeEqual(jraph.get_node_padding_mask(graph), array([True, True, True, False]))
        self.assertPytreeEqual(jraph.get_edge_padding_mask(graph), array([True, True, True, True, True, True]))
        self.assertPytreeEqual(jraph.get_graph_padding_mask(graph), array([True, False]))
        self.assertEqual(jraph.get_number_of_padding_with_graphs_nodes(graph), 1)
        self.assertEqual(jraph.get_number_of_padding_with_graphs_edges(graph), 0)
        self.assertEqual(jraph.get_number_of_padding_with_graphs_graphs(graph), 1)

    def test_001_make_graph_with_radius(self):
        nodes = array([0, 1, 2])
        graph = make_neighbors_graph(nodes, 2, 1)
        self.assertPytreeEqual(graph.nodes, array([[0, 1, 2, 0]]).T)  # last node is a padding node
        self.assertPytreeEqual(graph.edges, array([[1, -1, 1, -1, 0, 0]]).T)
        self.assertPytreeEqual(jraph.get_node_padding_mask(graph), array([True, True, True, False]))
        self.assertPytreeEqual(jraph.get_edge_padding_mask(graph), array([True, True, True, True, False, False]))
        self.assertPytreeEqual(jraph.get_graph_padding_mask(graph), array([True, False]))
        self.assertEqual(jraph.get_number_of_padding_with_graphs_nodes(graph), 1)
        self.assertEqual(jraph.get_number_of_padding_with_graphs_edges(graph), 2)
        self.assertEqual(jraph.get_number_of_padding_with_graphs_graphs(graph), 1)

    def test_002_make_grid_graph(self):
        nodes = array([0, 1, 5, 10])
        graph = make_grid_graph(nodes)
        self.assertPytreeEqual(graph.nodes, array([[0, 1, 5, 10, 0]]).T)  # last node is a padding node
        self.assertPytreeEqual(graph.edges, array([[5, -5, 4, -4, 1, -1, 0, 0]]).T)
        self.assertPytreeEqual(jraph.get_node_padding_mask(graph), array([True, True, True, True, False]))
        self.assertPytreeEqual(jraph.get_edge_padding_mask(graph), array([True, True, True, True, True, True, False, False]))
        self.assertPytreeEqual(jraph.get_graph_padding_mask(graph), array([True, False]))
        self.assertEqual(jraph.get_number_of_padding_with_graphs_nodes(graph), 1)
        self.assertEqual(jraph.get_number_of_padding_with_graphs_edges(graph), 2)
        self.assertEqual(jraph.get_number_of_padding_with_graphs_graphs(graph), 1)


class TestGraphIntegrationLayer(JaxTestCase):
    def test_000_graph_integration_layer(self):
        nodes = array([0, 1, 2, 3])
        weights = array([1 / 4] * 4)
        nodes = (weights, nodes)
        graph = make_neighbors_graph(nodes, 2)
        layer = GNNIntegrationLayer([5, 1])
        params = layer.init(random.key(0), graph)
        graph = layer.apply(params, graph)
        self.assertEqual(graph.globals.shape, (2, 1))

    def test_001_graph_integration_layer_without_weights(self):
        nodes = array([0, 1, 2, 3])
        graph = make_neighbors_graph(nodes, 2)
        layer = GNNIntegrationLayer([5, 1])
        params = layer.init(random.key(0), graph)
        graph = layer.apply(params, graph)
        self.assertEqual(graph.globals.shape, (2, 1))

    def test_002_graph_integration_layer_from_grid(self):
        nodes = array([0, 1, 2, 3])
        graph = make_grid_graph(nodes)
        layer = GNNIntegrationLayer([5, 1])
        params = layer.init(random.key(0), graph)
        graph = layer.apply(params, graph)
        self.assertEqual(graph.globals.shape, (2, 1))
        self.assertEqual(graph.globals[1, 0], 0)
        self.assertNotEqual(graph.globals[0, 0], 0)

    def test_002_graph_integration_layer_from_grid_with_weights(self):
        nodes = array([0, 1, 2, 3])
        weights = array([1 / 4] * 4)
        nodes = (weights, nodes)
        graph = make_grid_graph(nodes)
        layer = GNNIntegrationLayer([5, 1])
        params = layer.init(random.key(0), graph)
        graph = layer.apply(params, graph)
        self.assertEqual(graph.globals.shape, (2, 1))
        self.assertEqual(graph.globals[1, 0], 0)
        self.assertNotEqual(graph.globals[0, 0], 0)


class TestGraphLayer(JaxTestCase):
    def test_000_graph_layer(self):
        nodes = array([0, 1, 2, 3])
        graph = make_grid_graph(nodes)
        layer = GNNLayer([5, 2], [5, 2])
        params = layer.init(random.key(0), graph)
        graph = layer.apply(params, graph)

        self.assertEqual(graph.nodes.shape, (5, 2))
        self.assertNotEqual(graph.nodes[0, 0], 0)
        self.assertPytreeEqual(graph.nodes[-1], zeros((2,)))  # test if padding node is zero

        self.assertEqual(graph.edges.shape, (8, 2))
        self.assertNotEqual(graph.edges[0, 0], 0)
        self.assertPytreeEqual(graph.edges[-1], zeros((2,)))  # test if last padding edge is zero
        self.assertIsNone(graph.globals)

    def test_001_graph_layer_with_weights(self):
        nodes = array([0, 1, 2, 3])
        weights = array([1 / 4] * 4)
        nodes = (weights, nodes)
        graph = make_grid_graph(nodes)
        layer = GNNLayer([5, 2], [5, 2])
        params = layer.init(random.key(0), graph)
        graph = layer.apply(params, graph)
        
        self.assertEqual(graph.nodes[1].shape, (5, 2))
        self.assertNotEqual(graph.nodes[1][0, 0], 0)
        self.assertPytreeEqual(graph.nodes[1][-1], zeros((2,)))  # test if padding node is zero

        self.assertEqual(graph.edges.shape, (8, 2))
        self.assertNotEqual(graph.edges[0, 0], 0)
        self.assertPytreeEqual(graph.edges[-1], zeros((2,)))  # test if last padding edge is zero
        self.assertIsNone(graph.globals)