"""Tests for the functions and classes in the line module.
"""

import unittest

import networkx as nx
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from line import AliasTable, create_line_model, LINE, LINEPyDataset, line_loss


class TestAliasTable(unittest.TestCase):
    num_samples = 100_000
    rng = np.random.default_rng(42)


    def test_return_indices(self):
        weights = [0.2, 0.2, 0.4, 0.2]
        table = AliasTable(weights=weights)
        idx, counts = np.unique(table.sample(size=self.num_samples, rng=self.rng), return_counts=True)

        assert np.all(np.round(counts / np.sum(counts), 1) == np.array(weights))
        assert np.all(idx == np.arange(len(weights)))


    def test_return_labels(self):
        weights = [0.2, 0.2, 0.4, 0.2]
        labels = ["a", "b", "c", "d"]
        table = AliasTable(weights=weights, labels=labels)
        idx, counts = np.unique(table.sample(size=self.num_samples, rng=self.rng), return_counts=True)

        idx_labels = np.argsort(idx)

        counts = counts[idx_labels]
        idx = idx[idx_labels]

        assert np.all(np.round(counts / np.sum(counts), 1) == np.array(weights))
        assert np.all(idx == np.array(labels))


    def test_uniform(self):
        weights = [0.25, 0.25, 0.25, 0.25]
        table = AliasTable(weights=weights)
        idx, counts = np.unique(table.sample(size=self.num_samples, rng=self.rng), return_counts=True)

        assert np.all(np.round(counts / np.sum(counts), 2) == np.array(weights))
        assert np.all(idx == np.arange(len(weights)))


class BaseGraphTestCase(unittest.TestCase):
    graph = nx.Graph()
    graph.add_weighted_edges_from([
        ("0", "1", 1),
        ("0", "2", 1),
        ("0", "3", 1),
        ("4", "1", 1),
        ("4", "2", 1),
        ("4", "3", 1),
    ])
    rng = np.random.default_rng(2024)
    batch_size = 12


class TestLINEPyDataset(BaseGraphTestCase):
    def test_dataset_alias_table_uniform(self):
        dataset = LINEPyDataset(self.graph, batch_size=self.batch_size, rng=self.rng)

        assert len(dataset) == self.graph.number_of_edges() * \
             (1 + dataset.negative_ratio) / self.batch_size

        batch_x = [dataset[i][0] for i in range(len(dataset))]
        batch_y = [dataset[i][1] for i in range(len(dataset))]

        assert np.all(batch_x[0][0] == 0)
        assert np.all(np.isin(batch_x[1][0], [0, 1]))
        assert np.all(np.isin(batch_x[2][0], [2, 3]))
        assert np.all(np.isin(batch_x[0][1][:2], [1, 2]))

        assert np.all(np.abs(batch_y) == 1)
        assert np.all(batch_y[0][0][:2] == 1)
        assert np.all(batch_y[0][0][3:] == -1)


    def test_dataset_alias_table_not_uniform(self):
        graph = nx.Graph()
        graph.add_weighted_edges_from([
            ("0", "1", 1),
            ("0", "2", 1),
            ("0", "3", 1),
            ("4", "1", 1),
            ("4", "2", 1),
            ("4", "3", 100_000),
        ])
        dataset = LINEPyDataset(graph, batch_size=self.batch_size, rng=self.rng)

        batch_x = [dataset[i][0] for i in range(len(dataset))]
        batch_y = [dataset[i][1] for i in range(len(dataset))]

        assert np.all(batch_x[0][0] == 3)
        assert np.all(batch_x[0][1][:2] == 4)
        assert np.all(batch_x[0][1][3:] == 3)

        assert np.all(np.abs(batch_y) == 1)
        assert np.all(batch_y[0][0][:2] == 1)
        assert np.all(batch_y[0][0][3:] == -1)


    def test_dataset_two_outputs(self):
        dataset = LINEPyDataset(self.graph, batch_size=self.batch_size,
            two_outputs=True, rng=self.rng)

        batch_y = [dataset[i][1] for i in range(len(dataset))]

        assert np.all(np.abs(batch_y) == 1)
        assert np.all(batch_y[0][0][:2] == 1)
        assert np.all(batch_y[0][0][3:] == -1)
        assert np.all(batch_y[0][1][:2] == 1)
        assert np.all(batch_y[0][1][3:] == -1)


    def test_dataset_uneven_batch(self):
        dataset = LINEPyDataset(self.graph, batch_size=16,
            two_outputs=True, rng=self.rng)

        batch_x = [dataset[i][0] for i in range(len(dataset))]
        batch_y = [dataset[i][1] for i in range(len(dataset))]

        assert np.all(np.isin(batch_x[1][0], [0, 1]))
        assert np.all(np.isin(batch_x[2][0], [2, 3]))
        assert np.all(np.isin(batch_x[0][1][:2], [1, 2]))

        assert np.all(np.abs(batch_y) == 1)
        assert np.all(batch_y[0][0][:2] == 1)
        assert np.all(batch_y[0][0][2:] == -1)


class TestCreateModel(BaseGraphTestCase):
    def test_create_model(self):
        dataset = LINEPyDataset(self.graph, batch_size=self.batch_size, rng=self.rng)

        model, _ = create_line_model(numNodes=5, embedding_dim=2)

        model.compile("adam", line_loss)

        _ = model.fit(dataset, epochs=1)

        assert True


class TestLINE(BaseGraphTestCase):
    def test_line_first_order(self):
        graph = nx.Graph()
        graph.add_weighted_edges_from([
            ("0", "1", 1),
            ("0", "2", 1),
            ("0", "3", 1),
            ("0", "4", 1),
            ("0", "5", 1),
            ("0", "6", 1),
        ])
        dataset = LINEPyDataset(graph, batch_size=self.batch_size, rng=self.rng)

        model = LINE(dataset.node_size, embedding_dim=2, order="second")

        _ = model.train(dataset, epochs=50)

        embeddings_dict = model.get_embeddings(dataset.idx2node)

        embeddings = np.array(list(embeddings_dict.values()))

        similarity = cosine_similarity(embeddings)

        assert True

if __name__ == "__main__":
    unittest.main()
