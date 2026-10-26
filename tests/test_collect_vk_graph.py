"""Проверки сбора и экспорта графа на искусственных ответах API, без обращений к VK."""
import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs

import collect_vk_graph as collect


class CollectorTests(unittest.TestCase):
    """Проверяет получение списков, работу кэша и границы экспортируемого графа."""

    def setUp(self):
        """Создаёт временную папку, чтобы каждый тест работал со своими файлами."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def rows(self, name):
        """Читает строки результата из временной папки как словари по заголовкам CSV."""
        with (self.directory / name).open(newline="") as file:
            return list(csv.DictReader(file))

    def run_main(self, client, *arguments):
        """Запускает сборщик с имитацией VK API и файлами во временной папке.

        Настоящих запросов к VK нет: ответы задаёт тестовый client.
        Вывод перехватывается, чтобы сообщения сборщика не мешали читать результат теста.
        """
        (self.directory / "seeds.csv").write_text("group,profile\n1,1\n")
        command = ["collect_vk_graph.py", "--seeds", str(self.directory / "seeds.csv"),
                   "--out", str(self.directory), *arguments]
        with patch.dict(os.environ, VK_TOKEN="test-token"), patch("sys.argv", command), \
                patch.object(collect, "VKClient", return_value=client), \
                redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            return collect.main()

    def test_profile_formats_and_group_memberships(self):
        """Проверяет разбор ссылок и ID, сохранение групп и однократное разрешение имени."""
        file = self.directory / "seeds.csv"
        file.write_text("# comment\ngroup,profile\n1,https://m.vk.ru/example\n2,example\n1,id42\n1,id42\n")
        records = collect.load_seeds(file)
        client = Mock()
        client.call_api.return_value = [{"id": 42, "is_closed": True}]
        self.assertEqual(collect.resolve_seeds(client, records), {42: {"1", "2"}})
        client.call_api.assert_called_once_with("users.get", {"user_ids": "example"})
        for invalid in ("https://example.com/user", "https://vk.ru/name/other", "0", ""):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                collect.normalize_profile(invalid)

    def test_transport_posts_token_once_and_applies_fixed_delay_after_error(self):
        """Проверяет передачу токена через stdin, отсутствие повторов и интервал после ошибки."""
        replies = [SimpleNamespace(returncode=0, stdout=json.dumps({"error": {"error_code": 9, "error_msg": "Flood control"}})),
                   SimpleNamespace(returncode=0, stdout='{"response": []}')]
        with patch.object(collect.subprocess, "run", side_effect=replies) as run, \
                patch.object(collect.time, "monotonic", side_effect=[10, 10.25, 11]), \
                patch.object(collect.time, "sleep") as sleep:
            client = collect.VKClient("private-token", 1)
            with self.assertRaises(collect.VKError) as raised:
                client.call_api("friends.get", {"user_id": 42})
            self.assertEqual(raised.exception.code, 9)
            self.assertEqual(run.call_count, 1)
            client.call_api("friends.get", {"user_id": 43})
        sleep.assert_called_once_with(0.75)
        for call in run.call_args_list:
            self.assertNotIn("private-token", " ".join(call.args[0]))
            self.assertEqual(parse_qs(call.kwargs["input"])["access_token"], ["private-token"])

    def test_get_friends_paginates_and_rejects_incomplete_results(self):
        """Проверяет загрузку страниц и отказ считать оборванную выдачу полным списком."""
        client = Mock()
        client.call_api.side_effect = [{"count": 3, "items": [2, 3]}, {"count": 3, "items": [4]}]
        self.assertEqual(collect.get_friends(client, 1), [2, 3, 4])
        self.assertEqual(client.call_api.call_args.args[1]["offset"], 2)
        client.call_api.side_effect = [{"count": 3, "items": [2]}, {"count": 3, "items": []}]
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            collect.get_friends(client, 1)

    def test_cache_is_saved_before_next_profile_including_empty_lists(self):
        """Проверяет сохранение кэша и пропуск уже полученных, в том числе пустых, списков."""
        cache = {"1": [2, 3], "2": []}
        cache_path = self.directory / "friends_cache.json"

        def response(method, params):
            """Подставляет пустой список и проверяет, что запрос сделан только для нового профиля."""
            self.assertEqual(params["user_id"], 3)
            return {"count": 0, "items": []}

        client = Mock()
        client.call_api.side_effect = response
        with redirect_stderr(io.StringIO()):
            collect.collect_friends(client, {1: {"1"}}, cache, cache_path, [], 2, 0)
            collect.collect_friends(client, {1: {"1"}}, cache, cache_path, [], 2, 0)
        self.assertEqual(client.call_api.call_count, 1)
        self.assertEqual(json.loads(cache_path.read_text()), {"1": [2, 3], "2": [], "3": []})

    def test_privacy_error_is_reported_without_precheck_or_retry(self):
        """Проверяет запись ошибки закрытого профиля и продолжение сбора без проверки и повторов."""
        client = Mock()
        client.call_api.side_effect = [{"count": 2, "items": [2, 3]}, collect.VKError(30, "Private profile"), {"count": 0, "items": []}]
        self.assertEqual(self.run_main(client), 0)
        self.assertEqual([call.args[0] for call in client.call_api.call_args_list], ["friends.get"] * 3)
        self.assertEqual(len(self.rows("errors.csv")), 1)
        self.assertEqual(json.loads((self.directory / "friends_cache.json").read_text()), {"1": [2, 3], "3": []})
        self.assertFalse((self.directory / "profile_status.json").exists())
        self.assertFalse((self.directory / "api_state.json").exists())

    def test_flood_stops_and_preserves_completed_lists_and_graph(self):
        """Проверяет остановку при флуде с сохранением уже полученных списков и частичного графа."""
        client = Mock()
        client.call_api.side_effect = [{"count": 2, "items": [2, 3]}, collect.VKError(9, "Flood control")]
        self.assertEqual(self.run_main(client), 2)
        self.assertEqual(client.call_api.call_count, 2)
        self.assertEqual(json.loads((self.directory / "friends_cache.json").read_text()), {"1": [2, 3]})
        self.assertEqual({row["id"] for row in self.rows("nodes.csv")}, {"1", "2", "3"})
        self.assertEqual(len(self.rows("edges.csv")), 2)

    def test_failed_later_page_is_not_cached(self):
        """Проверяет, что ошибка следующей страницы не оставляет в кэше неполный список."""
        client = Mock()
        client.call_api.side_effect = [{"count": 3, "items": [2]}, collect.VKError(9, "Flood control")]
        self.assertEqual(self.run_main(client), 2)
        self.assertEqual(json.loads((self.directory / "friends_cache.json").read_text()), {})

    def test_interrupt_saves_partial_graph(self):
        """Проверяет сохранение полученных данных при прерывании через Ctrl+C."""
        client = Mock()
        client.call_api.side_effect = [{"count": 1, "items": [2]}, KeyboardInterrupt]
        self.assertEqual(self.run_main(client), 130)
        self.assertEqual(json.loads((self.directory / "friends_cache.json").read_text()), {"1": [2]})
        self.assertEqual(len(self.rows("edges.csv")), 1)

    def test_boundary_pass_fetches_second_hop_but_never_adds_third_hop(self):
        """Проверяет запрос списков второго круга без добавления вершин третьего круга."""
        client = Mock()
        lists = {1: [2], 2: [1, 3], 3: [2, 4]}
        client.call_api.side_effect = lambda method, params: {"count": len(lists[params["user_id"]]), "items": lists[params["user_id"]]}
        self.assertEqual(self.run_main(client, "--complete-boundary"), 0)
        self.assertEqual([call.args[1]["user_id"] for call in client.call_api.call_args_list], [1, 2, 3])
        self.assertEqual({row["id"] for row in self.rows("nodes.csv")}, {"1", "2", "3"})

    def test_default_only_queries_seeds_and_first_hop(self):
        """Проверяет, что обычный сбор ограничивается запросами исходных профилей и первого круга."""
        client = Mock()
        client.call_api.side_effect = [{"count": 1, "items": [2]}, {"count": 2, "items": [1, 3]}]
        self.assertEqual(self.run_main(client), 0)
        self.assertEqual(client.call_api.call_count, 2)
        self.assertEqual({row["id"] for row in self.rows("nodes.csv")}, {"1", "2", "3"})

    def test_export_uses_all_cached_paths_and_deduplicates_edges(self):
        """Проверяет экспорт по всему кэшу, удаление петель и повторов, сохранение учебных групп."""
        cache = {"1": [1, 2, 2, 3], "2": [1, 4], "3": [1, 5], "4": [2, 5, 6], "5": []}
        self.assertEqual(collect.export_graph(self.directory, {1: {"1", "2"}}, cache), (5, 5))
        pairs = [(int(row["source"]), int(row["target"])) for row in self.rows("edges.csv")]
        self.assertEqual(set(pairs), {(1, 2), (1, 3), (2, 4), (3, 5), (4, 5)})
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(self.rows("nodes.csv")[0]["seed_groups"], "1;2")


if __name__ == "__main__":
    unittest.main()
