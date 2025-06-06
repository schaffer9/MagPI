
from typing import Callable
import itertools

import jraph
from jaxkd import build_tree, query_neighbors

from .prelude import *
from .model import MLP


class GNNIntegrationLayer(nn.Module):
    """This layer updates the globals of the graph, by integration. Therefore integration weights on the nodes
    need to be provided. This can be done by providing a tuple for the nodes `(node_weights, node_features)`.
    If no such tuple is provided, integration is performed with Monte Carlo.

    After integration, the globals are transformed with a MLP with the architecture given by `globals_mlp`.
    
    """
    globals_mlp: list[int]
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        assert graph.nodes is not None
        nodes = graph.nodes
        graph_mask = asarray(jraph.get_graph_padding_mask(graph))

        if isinstance(graph.nodes, Array):
            # if no weights are provided we perform MC integration with weights 1 / N
            aggregate_nodes_for_globals_fn = jraph.segment_mean
            update_node_fn = lambda node_features, s, r, g: node_features

        else:
            aggregate_nodes_for_globals_fn = jraph.segment_sum
            def _update_node_fn(node_features, aggregated_sender_edge_features, aggregated_receiver_edge_features, globals_):
                weights, nodes = node_features
                weighted_nodes = tree.map(lambda n: n * weights[:, *[None for _ in n.shape[1:]]], nodes)
                return weighted_nodes
            
            update_node_fn = _update_node_fn

        update_edge_fn = lambda e, s, r, g: e
        
        def update_globals_fn(integrated_node_features, aggregated_edge_features, globals_):
            if globals_ is not None:
                x = jnp.concatenate([integrated_node_features, globals_], -1)
            else:
                x = integrated_node_features
            
            x = MLP(self.globals_mlp, self.activation)(x)
            return x
        
        network = jraph.GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            update_global_fn=update_globals_fn,
            aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn)

        graph = network(graph)
        globals_ = tree.map(lambda g: where(graph_mask[:, *[None for _ in g.shape[1:]]], g, 0), graph.globals)  # set padding to zero
        graph = graph._replace(nodes=nodes, globals=globals_)  # replace weighted nodes with old nodes
        return graph


class GNNLayer(nn.Module):
    update_edge_mlp: list[int]
    update_node_mlp: list[int]
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        nodes = graph.nodes
        node_mask = jraph.get_node_padding_mask(graph)
        edge_mask = jraph.get_edge_padding_mask(graph)
        if not isinstance(graph.nodes, Array):
            assert graph.nodes is not None
            weights, nodes = graph.nodes
            graph = graph._replace(nodes=nodes)
        else:
            weights = None

        @jraph.concatenated_args
        def update_edge_fn(features):
            return MLP(self.update_edge_mlp, self.activation)(features)
        
        @jraph.concatenated_args
        def update_node_fn(features):
            return MLP(self.update_node_mlp, self.activation)(features)
        
        def update_global_fn(aggregated_node_features, aggregated_edge_features, globals_):
            return globals_

        graph = jraph.GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            update_global_fn=update_global_fn,
            aggregate_edges_for_nodes_fn=jraph.segment_mean
        )(graph)
        if weights is not None:
            graph = graph._replace(nodes=(weights, graph.nodes))

        nodes = tree.map(lambda g: where(node_mask[:, *[None for _ in g.shape[1:]]], g, 0), graph.nodes)
        edges = tree.map(lambda g: where(edge_mask[:, *[None for _ in g.shape[1:]]], g, 0), graph.edges)
        graph = graph._replace(nodes=nodes, edges=edges)
        return graph


def nearest_neighbors(
        nodes: Array,
        max_neighbors: int,
        radius: float | None = None) -> tuple[Array, Array]:
    """Finds nearest neighbors with an optional constraint on the distance.

    Parameters
    ----------
    nodes : Array
    max_neighbors : int
        maximum number of neighbors
    radius : float | None, optional
        only select neighbors falling within this radius, by default None

    Returns
    -------
    tuple[Array, Array]
        neighbors indices and padding mask 
        (`True` for real neighbors and `False` for padding connections)
    """
    if nodes.ndim == 1:
        nodes = nodes[:, None]
    kdtree = build_tree(nodes)
    neighbors, distances = query_neighbors(kdtree, nodes, k=max_neighbors + 1)
    neighbors, distances = neighbors[:, 1:], distances[:, 1:]  # remove self connections
    padding = jnp.full(neighbors.shape, True)
    if radius is None:
        return neighbors, padding
    else:
        padding = where(distances <= radius, True, False)
        neighbors = where(distances <= radius, neighbors, -1)
        return neighbors, padding


def grid_neighbors(grid: Array, dilation: int = 1):
    """Creates a graph based on the full Moore neighborhood on an 
    irregular grid.

    Parameters
    ----------
    grid : Array
        grid array with dimensions :math:`(n_1, n_2, ..., n_d, d)`
    dilation : int, optional
        dilation rate, by default 1

    Returns
    -------
    _type_
        _description_
    """
    if grid.ndim == 1:
        grid = grid[:, None]
    dims = grid.shape[:-1]
    d = grid.shape[-1]
    assert len(dims) == d, "Dimension error! Provide a regular grid!"
    offsets = asarray([
        offset for offset in itertools.product([-dilation, 0, dilation], repeat=d)
        if any(x != 0 for x in offset)  # excluding the center (0, 0, ..., 0)
    ])
    indices = jnp.stack(jnp.meshgrid(*[jnp.arange(_s) for _s in dims], indexing="ij"), axis=-1).reshape(-1, d)

    def neighbors_of(idx):
        neighbors = idx[None, :] + offsets
        assert neighbors.shape == (3 ** d - 1, d)
        # Check bounds
        valid = jnp.all(asarray([(neighbors[:, i] >= 0) & (neighbors[:, i] < n) for i, n in enumerate(dims)]), axis=0)
        # Convert to flat indices
        flat_neighbors = jnp.where(
            valid,
            vmap(lambda i: jnp.ravel_multi_index(i, dims, mode="clip"))(neighbors),
            -1  # padding index
        )

        return flat_neighbors, valid

    neighbor_indices, padding_mask = vmap(neighbors_of)(indices)
    return neighbor_indices, padding_mask


def _make_graph(nodes: Array, senders: Array, receivers: Array, padding: Array) -> jraph.GraphsTuple:
    assert nodes.ndim == 2
    n = nodes.shape[0]
    senders = jnp.where(padding, senders, n)  # padding connections go to the padding node
    receivers = jnp.where(padding, receivers, n)
    senders, receivers = senders.ravel(), receivers.ravel()
    nodes = jnp.append(nodes, zeros((1, nodes.shape[1])), 0)  # add a padding node
    edge_features = nodes[senders] - nodes[receivers]

    # sort edges so that padding edges come last
    d = norm(edge_features, axis=-1)
    idx = jnp.argsort(d, descending=True)
    edge_features = edge_features[idx]
    receivers = receivers[idx]
    senders = senders[idx]

    _k = jnp.count_nonzero(receivers != n)
    n_edge = jnp.asarray([_k, len(receivers) - _k])
    n_node = jnp.asarray([n, 1])

    graph = jraph.GraphsTuple(nodes=nodes, senders=senders, receivers=receivers,
                              edges=edge_features, n_node=n_node, n_edge=n_edge, globals=None)
    return graph


def make_neighbors_graph(nodes: Array | tuple[Array, Array], max_neighbors: int, radius: float | None = None):
    if isinstance(nodes, tuple):
        weights, nodes = nodes
    else:
        weights = None
    if nodes.ndim == 1:
        nodes = nodes[:, None]
    senders, padding = nearest_neighbors(nodes, max_neighbors, radius)
    receivers = jnp.repeat(
        jnp.arange(0, nodes.shape[0])[:, None], senders.shape[-1], axis=-1
    )
    graph = _make_graph(nodes, senders, receivers, padding)
    if weights is not None:
        weights = jnp.append(weights, zeros((1,)), axis=0)
        graph = graph._replace(nodes=(weights, graph.nodes))
    return graph

def make_grid_graph(grid: Array | tuple[Array, Array], dilation: int = 1):
    if isinstance(grid, tuple):
        weights, nodes = grid
    else:
        nodes = grid
        weights = None
    if nodes.ndim == 1:
        nodes = nodes[:, None]
    senders, padding = grid_neighbors(nodes, dilation)
    nodes = nodes.reshape(-1, nodes.shape[-1])
    receivers = jnp.repeat(
        jnp.arange(0, nodes.shape[0])[:, None], senders.shape[-1], axis=-1
    )
    graph = _make_graph(nodes, senders, receivers, padding)
    if weights is not None:
        weights = weights.ravel()
        weights = jnp.append(weights, zeros((1,)), axis=0)
        graph = graph._replace(nodes=(weights, graph.nodes))
    return graph