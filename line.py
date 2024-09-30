""" Implementation of the LINE: Large Scale Informed Network Embeddings.

Method developed by: Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J., & Mei, Q. (2015).
LINE: Large-scale information network embedding.
Proceedings of the 24th International Conference on World Wide Web, 1067–1077.
https://doi.org/10.1145/2736277.2741093

Implementation adapted from: Weichen Shen @ https://github.com/shenweichen/GraphEmbedding

"""

import argparse
import os
import warnings

os.environ["KERAS_BACKEND"] = "torch"

import gzip
import keras
import numpy as np
import networkx as nx


def line_loss(y_true, y_pred):
    loss = -keras.ops.mean(keras.ops.log(keras.ops.sigmoid(y_true * y_pred)))
    return loss


class AliasTable:
    def __init__(self, weights, labels=None):
        self.weights = np.array(weights)

        if labels is not None:
            self.labels = np.array(labels)
        else:
            self.labels = None

        self.n = len(weights)

        self.accept, self.alias = self._create_table(self.weights, self.n)

    
    @staticmethod
    def _create_table(weights, n):
        accept, alias = [0] * n, [0] * n
        small, large = [], []
        weights_ = weights * n
        for i, prob in enumerate(weights_):
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            small_idx, large_idx = small.pop(), large.pop()
            accept[small_idx] = weights_[small_idx]
            alias[small_idx] = large_idx
            weights_[large_idx] = weights_[large_idx] - (1 - weights_[small_idx])

            if weights_[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)

        while large:
            large_idx = large.pop()
            accept[large_idx] = 1
        while small:
            small_idx = small.pop()
            accept[small_idx] = 1

        return np.array(accept), np.array(alias)

    
    def sample(self, size=1, idx=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        if idx is not None:
            size = len(idx)
        else:
            idx = (rng.uniform(size=size) * self.n).astype(np.int64)

        probs = rng.uniform(size=size)
        
        out = np.where(probs < self.accept[idx], idx, self.alias[idx])

        if self.labels is not None:
            return self.labels[out]

        return out


def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def create_line_model(numNodes, embedding_dim, order='all'):
    v_i = keras.layers.Input(shape=(1,))
    v_j = keras.layers.Input(shape=(1,))

    first_emb = keras.layers.Embedding(numNodes, embedding_dim, name='first_emb')
    second_emb = keras.layers.Embedding(numNodes, embedding_dim, name='second_emb')
    context_emb = keras.layers.Embedding(numNodes, embedding_dim, name='context_emb')

    v_i_emb = first_emb(v_i)
    v_j_emb = first_emb(v_j)

    v_i_emb_second = second_emb(v_i)
    v_j_context_emb = context_emb(v_j)

    first_dot = keras.layers.Lambda(lambda x: keras.ops.sum(
        x[0] * x[1], axis=-1, keepdims=False), name='first_order')
    second_dot = keras.layers.Lambda(lambda x: keras.ops.sum(
        x[0] * x[1], axis=-1, keepdims=False), name='second_order')

    first = first_dot([v_i_emb, v_j_emb])
    second = second_dot([v_i_emb_second, v_j_context_emb])

    if order == 'first':
        output_list = [first]
    elif order == 'second':
        output_list = [second]
    else:
        output_list = [first, second]

    model = keras.Model(inputs=[v_i, v_j], outputs=output_list)

    return model, {'first': first_emb, 'second': second_emb}


class LINEPyDataset(keras.utils.PyDataset):
    def __init__(self, graph, batch_size, negative_ratio=5, power=0.75, two_outputs=False, rng=None, **kwargs):
        super().__init__(**kwargs)
        if batch_size < 1 + negative_ratio:
            raise ValueError("Batch size must be higher than (1 + negative ratio)")

        self.graph = graph

        remainder_batch_size = batch_size % (1 + negative_ratio)

        if remainder_batch_size > 0:
            batch_size = (batch_size // (1 + negative_ratio)) * (1 + negative_ratio)
            warnings.warn(f"Batch size not multiple of (1 + negative ratio): Using batch size {batch_size} instead")

        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.power = power
        self.two_outputs = two_outputs

        if rng is None:
            rng = np.random.default_rng()
        
        self.rng = rng

        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        self.samples_per_epoch = self.edge_size * (1 + negative_ratio)
        self.steps_per_epoch = ((self.samples_per_epoch - 1) // self.batch_size + 1)
        self.pos_size = self.batch_size // (1 + negative_ratio)
        self.neg_size = self.pos_size * negative_ratio

        out_degree = np.zeros(self.node_size)

        for edge in self.graph.edges():
            out_degree[self.node2idx[edge[0]]
            ] += self.graph[edge[0]][edge[1]].get('weight', 1.0)

        out_degree_pow = out_degree ** self.power

        node_norm_prob = out_degree_pow / np.sum(out_degree_pow)

        self.node_alias_table = AliasTable(node_norm_prob)

        edge_weights = np.array([self.graph[e[0]][e[1]].get("weight", 1.0) for e in self.graph.edges()])
        edge_norm_prob = edge_weights / np.sum(edge_weights)

        self.edge_alias_table = AliasTable(edge_norm_prob)

        self.edges_idx = np.array([(self.node2idx[x[0]], self.node2idx[x[1]]) for x in self.graph.edges()])


    def __len__(self):
        return self.steps_per_epoch


    def __getitem__(self, idx):
        start_idx = idx * self.pos_size
        end_idx = min(start_idx + self.pos_size, self.edge_size)

        batch_edge_idx = self.edge_alias_table.sample(idx=np.arange(start_idx, end_idx), rng=self.rng)

        batch_edges = np.array(self.edges_idx[batch_edge_idx])

        batch_source = batch_edges[:, 0]
        batch_target = batch_edges[:, 1]

        batch_neg = self.node_alias_table.sample(size=batch_edge_idx.shape[0] * self.negative_ratio, rng=self.rng)

        batch_sign = np.ones(batch_edge_idx.shape[0] * (1+self.negative_ratio))
        batch_sign[self.pos_size:] *= -1

        batch_x = [np.tile(batch_source, 1+self.negative_ratio), np.hstack([batch_target, batch_neg])]
        batch_y = [batch_sign]

        if self.two_outputs:
            batch_y += [batch_sign]

        return batch_x, batch_y


class LINE:
    def __init__(self, num_nodes, embedding_dim=8, order='second'):
        if order.lower() not in ["first", "second", "all"]:
            raise ValueError("Order must be 'first', 'second', or 'all'")

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.order = order
        self._embeddings = None

        self.model, self.embedding_dict = create_line_model(
            self.num_nodes, self.embedding_dim, self.order
        )

        self.model.compile("adam", line_loss)


    def get_embeddings(self, idx2node):
        if self._embeddings is not None:
            return self._embeddings

        self._embeddings = {}

        if self.order == 'first':
            embeddings = self.embedding_dict['first'].get_weights()[0]
        elif self.order == 'second':
            embeddings = self.embedding_dict['second'].get_weights()[0]
        else:
            embeddings = np.hstack((self.embedding_dict['first'].get_weights()[
                                        0], self.embedding_dict['second'].get_weights()[0]))

        for i, embedding in enumerate(embeddings):
            self._embeddings[idx2node[i]] = embedding

        return self._embeddings


    def train(self, dataset, epochs=1, shuffle=True, verbose=1):

        hist = self.model.fit(
            dataset,
            epochs=epochs,
            shuffle=shuffle,
            verbose=verbose
        )

        return hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--emb-dim", default=128, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--order", default="second")
    parser.add_argument("--negative-ratio", default=5, type=int)
    parser.add_argument("--no-shuffle", action="store_false")
    parser.add_argument("--verbose", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    graph = nx.read_weighted_edgelist(args.input)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    dataset = LINEPyDataset(
        graph,
        args.batch_size,
        args.negative_ratio,
        two_outputs=args.order == "all",
        rng=np.random.default_rng(args.seed)
    )

    model = LINE(dataset.node_size, embedding_dim=args.emb_dim, order=args.order)

    _ = model.train(dataset, epochs=args.epochs, shuffle=args.no_shuffle, verbose=args.verbose)

    embeddings = model.get_embeddings(dataset.idx2node)

    with gzip.open(args.output, 'wt') as file:
        for key, val in embeddings.items():
            file.write(str(key))
            for d in val:
                file.write(' ')
                file.write('%.8f'%d)
            file.write('\n')


if __name__ == "__main__":
    main()