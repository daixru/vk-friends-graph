"""Проверки центральностей на известных графах и формирования рейтингов из CSV."""
import csv
import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import igraph as ig
import numpy as np

import analyze_graph as analysis
from reports import export_top_rankings


class AnalysisTests(unittest.TestCase):
    """Сравнивает расчёты с известными значениями и проверяет выгрузку результатов."""

    def setUp(self):
        """Создаёт отдельную временную папку для входных и выходных файлов каждого теста."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def write_csv(self, name, rows):
        """Записывает тестовую CSV-таблицу и возвращает путь к ней."""
        file = self.directory / name
        with file.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        return file

    def rows(self, file):
        """Читает CSV как список записей, чтобы сравнивать значения по именам колонок."""
        with file.open(newline="") as stream:
            return list(csv.DictReader(stream))

    def scores(self, graph, samples=100):
        """Считает три показателя тестового графа с seed=42, скрывая сообщения прогресса."""
        with redirect_stderr(io.StringIO()):
            return (analysis.native_betweenness(graph, samples, 42),
                    analysis.native_closeness(graph, list(graph.connected_components()), samples, 42),
                    analysis.native_eigenvector(graph, list(graph.connected_components())))

    def test_exact_path_has_known_centralities(self):
        """Проверяет точные показатели цепочки из трёх вершин по значениям, вычисляемым вручную."""
        graph = ig.Graph(n=3, edges=[(0, 1), (1, 2)])
        between, close, eigen = self.scores(graph)
        np.testing.assert_allclose(between, [0, 1, 0])
        np.testing.assert_allclose(close, [2 / 3, 1, 2 / 3])
        np.testing.assert_allclose(eigen, [0.5, math.sqrt(0.5), 0.5])

    def test_disconnected_graph_uses_wf_and_component_eigenvectors(self):
        """Проверяет поправку близости и отдельную нормировку собственного вектора в компонентах."""
        graph = ig.Graph(n=4, edges=[(0, 1), (1, 2)])
        between, close, eigen = self.scores(graph)
        np.testing.assert_allclose(between, [0, 1 / 3, 0, 0])
        np.testing.assert_allclose(close, [4 / 9, 2 / 3, 4 / 9, 0])
        np.testing.assert_allclose(eigen, [0.5, math.sqrt(0.5), 0.5, 1])

    def test_sampling_scales_source_contributions(self):
        """Проверяет масштабирование выборочных вкладов посредничества на n/k."""
        graph = ig.Graph(n=3, edges=[(0, 1), (1, 2)])
        # seed=42 выбирает оба конца цепочки: выборочная оценка может превышать точную.
        between, _, _ = self.scores(graph, samples=2)
        np.testing.assert_allclose(between, [0, 1.5, 0])

    def test_sampled_closeness_on_clique_and_small_graphs(self):
        """Проверяет выборочную близость полного графа и особые случаи графов из одной-двух вершин."""
        between, close, _ = self.scores(ig.Graph.Full(8), samples=2)
        np.testing.assert_allclose(between, np.zeros(8))
        np.testing.assert_allclose(close, np.ones(8))
        for size in (1, 2):
            between, close, eigen = self.scores(ig.Graph(n=size))
            np.testing.assert_allclose(between, np.zeros(size))
            np.testing.assert_allclose(close, np.zeros(size))
            np.testing.assert_allclose(eigen, np.ones(size))

    def test_loader_handles_large_ids_duplicate_edges_loops_and_isolates(self):
        """Проверяет большие ID, сохранение изолированных вершин, удаление петель и повторных рёбер."""
        nodes = self.write_csv("nodes.csv", [("id",), (6000000000,), (2,), (3,)])
        edges = self.write_csv("edges.csv", [("source", "target"), (6000000000, 2), (2, 6000000000), (2, 2)])
        graph, ids = analysis.load_native_graph(nodes, edges)
        self.assertEqual(ids.tolist(), [6000000000, 2, 3])
        self.assertEqual((graph.vcount(), graph.ecount()), (3, 1))

    def test_cli_exact_writes_all_nodes_metadata_and_tops_without_checkpoints(self):
        """Проверяет полный запуск точного расчёта: CSV всех вершин, метаданные и топы без кэша этапов."""
        nodes = self.write_csv("nodes.csv", [("id",), (1,), (2,), (3,)])
        edges = self.write_csv("edges.csv", [("source", "target"), (1, 2), (2, 3)])
        output = self.directory / "centrality.csv"
        command = ["analyze_graph.py", "--nodes", str(nodes), "--edges", str(edges), "--out", str(output), "--exact", "--top", "2"]
        with patch("sys.argv", command), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(analysis.main(), 0)
        self.assertEqual(len(self.rows(output)), 3)
        self.assertTrue(json.loads(output.with_suffix(".meta.json").read_text())["exact"])
        self.assertEqual(self.rows(self.directory / "centrality_top2_betweenness.csv")[0]["id"], "2")
        self.assertFalse((self.directory / "centrality_cache").exists())

    def test_top_is_sorted_with_id_ties_and_metadata_not_filename(self):
        """Проверяет порядок при равенстве оценок и определение точности по метаданным."""
        source = self.write_csv("centrality_exact.csv", [("id", "betweenness", "closeness", "eigenvector"),
                                                          (9, 0.2, 0.1, 0.4), (3, 0.2, 0.3, 0.2), (5, 0.1, 0.2, 0.9)])
        source.with_suffix(".meta.json").write_text(json.dumps({"vertex_count": 3, "betweenness_method": "uniform source sampling, scaled by n/k",
                                                                "closeness_method": "exact WF", "eigenvector_method": "native", "component_count": 2}))
        with redirect_stderr(io.StringIO()):
            report = export_top_rankings(source, 2)
        between = self.rows(self.directory / "centrality_exact_top2_betweenness.csv")
        self.assertEqual([row["id"] for row in between], ["3", "9"])
        self.assertEqual(between[0]["method"], "approximate")
        self.assertEqual(self.rows(self.directory / "centrality_exact_top2_closeness.csv")[0]["method"], "exact_on_observed_graph")
        self.assertIn("несопоставимы", report.read_text())

    def test_rank_only_works_without_native_dependencies(self):
        """Проверяет построение топов без загрузки графа и без доступных igraph и NumPy."""
        source = self.write_csv("centrality.csv", [("id", "betweenness", "closeness", "eigenvector"), (1, 0, 0, 1)])
        with patch("sys.argv", ["analyze_graph.py", "--rank-only", "--out", str(source)]), \
                patch.object(analysis, "ig", None), patch.object(analysis, "np", None), \
                patch.object(analysis, "load_native_graph", side_effect=AssertionError("must not calculate")), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(analysis.main(), 0)
        self.assertEqual(self.rows(self.directory / "centrality_top10_eigenvector.csv")[0]["method"], "unknown")

    def test_reading_csv_can_change_access_time_without_invalidating_report(self):
        """Проверяет, что изменение времени доступа при чтении CSV не считается изменением данных."""
        source = self.write_csv("centrality.csv", [("id", "betweenness", "closeness", "eigenvector"), (1, 0, 0, 1)])
        os.utime(source, ns=(1, source.stat().st_mtime_ns))
        with redirect_stderr(io.StringIO()):
            self.assertTrue(export_top_rankings(source).is_file())

    def test_reports_reject_nonfinite_scores_and_inconsistent_metadata(self):
        """Проверяет отказ при NaN в метриках и несовпадении числа вершин с метаданными."""
        source = self.write_csv("bad.csv", [("id", "betweenness", "closeness", "eigenvector"), (1, "nan", 0, 1)])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            export_top_rankings(source)
        source = self.write_csv("bad.csv", [("id", "betweenness", "closeness", "eigenvector"), (1, 0, 0, 1)])
        source.with_suffix(".meta.json").write_text('{"vertex_count": 2}')
        with self.assertRaisesRegex(ValueError, "row count"):
            export_top_rankings(source)


if __name__ == "__main__":
    unittest.main()
