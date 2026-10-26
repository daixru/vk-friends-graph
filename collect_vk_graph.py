#!/usr/bin/env python3
"""Сбор списков друзей VK и построение графа в пределах двух кругов от исходных профилей."""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlencode, urlparse


class VKError(RuntimeError):
    """Ошибка VK API с числовым кодом для выбора дальнейшего действия при сборе."""

    def __init__(self, code: int, message: str):
        """Сохраняет код ошибки отдельно от сообщения, чтобы вызывающий код мог его проверить."""
        self.code = code
        super().__init__(f"VK error {code}: {message}")


def save_json(path: Path, value: dict) -> None:
    """Сохраняет JSON через временный файл и замену исходного файла.

    Старый кэш заменяется только после записи нового содержимого, чтобы
    не оставлять вместо него частично записанный JSON при прерывании записи.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class VKClient:
    """Клиент VK API с постоянным интервалом между запросами, без автоматических повторов."""

    def __init__(self, token: str, delay: float):
        """Задаёт токен и минимальный интервал между началами запросов.

        Время последнего запроса хранится только в памяти текущего запуска.
        """
        self.token = token
        self.delay = delay
        self.last_request = None

    def call_api(self, method: str, params: dict):
        """Один раз вызывает метод VK и возвращает содержимое поля response.

        Выдерживает заданный интервал, отправляет параметры через curl и разбирает
        JSON. Ошибки API передаёт как VKError, ошибки сети и ответа — как RuntimeError;
        решение о продолжении сбора принимает вызывающий код.
        """
        if self.last_request is not None:
            time.sleep(max(0, self.delay - (time.monotonic() - self.last_request)))
        # Токен передаём через stdin, чтобы он не попал в аргументы процесса curl.
        self.last_request = time.monotonic()
        try:
            result = subprocess.run(
                ["curl", "--silent", "--show-error", "--max-time", "30",
                 "--data-binary", "@-", f"https://api.vk.com/method/{method}"],
                input=urlencode({**params, "access_token": self.token, "v": "5.199"}),
                capture_output=True, text=True, timeout=35,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("Could not complete the VK request (curl/network failure)") from exc
        if result.returncode:
            raise RuntimeError(f"VK request failed (curl exit {result.returncode})")
        try:
            payload = json.loads(result.stdout)
            if "error" in payload:
                error = payload["error"]
                raise VKError(int(error["error_code"]), error["error_msg"])
            return payload["response"]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("VK returned an invalid response") from exc


def normalize_profile(value: str) -> str:
    """Приводит ссылку, короткое имя или запись id123 к идентификатору профиля.

    Для ссылки проверяет домен VK; числовой ID возвращает строкой без префикса id.
    Числовой ID по короткому имени здесь не запрашивается: это задача resolve_seeds.
    """
    value = value.strip()
    if "://" in value:
        parsed = urlparse(value)
        if parsed.netloc.lower() not in {"vk.ru", "www.vk.ru", "m.vk.ru", "vk.com", "www.vk.com", "m.vk.com"}:
            raise ValueError(f"Unsupported VK URL: {value}")
        value = unquote(parsed.path.strip("/"))
    if value.startswith("id") and value[2:].isdigit():
        value = value[2:]
    if not re.fullmatch(r"[A-Za-z0-9_.]+", value) or (value.isdigit() and int(value) <= 0):
        raise ValueError(f"Invalid VK profile: {value}")
    return value


def load_seeds(path: Path) -> list[tuple[str, str]]:
    """Читает исходные профили из CSV и возвращает пары «группа, профиль».

    Поддерживает две колонки group,profile и список из одной колонки.
    Пропускает пустые строки и комментарии, убирает повторы внутри одной группы.
    """
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = [row for row in csv.reader(file)
                if row and any(cell.strip() for cell in row) and not row[0].lstrip().startswith("#")]
    if rows and [cell.strip().lower() for cell in rows[0]] == ["group", "profile"]:
        rows = rows[1:]
    records = []
    for row in rows:
        if len(row) == 1:
            group, profile = "", row[0]
        elif len(row) == 2:
            group, profile = row
        else:
            raise ValueError("Seeds need one profile column, or group,profile columns")
        records.append((group.strip(), normalize_profile(profile)))
    return list(dict.fromkeys(records))


def resolve_seeds(client: VKClient, records: list[tuple[str, str]]) -> dict[int, set[str]]:
    """Возвращает соответствие числовых ID исходных профилей множествам их групп.

    Короткие имена разрешает через users.get, готовые ID использует сразу.
    В пределах вызова одно имя разрешается один раз, даже если оно есть в разных группах.
    """
    groups: dict[int, set[str]] = {}
    resolved = {}
    for group, profile in records:
        if profile not in resolved:
            if profile.isdigit():
                resolved[profile] = int(profile)
            else:
                users = client.call_api("users.get", {"user_ids": profile})
                if not users:
                    raise RuntimeError(f"Profile not found: {profile}")
                resolved[profile] = int(users[0]["id"])
        groups.setdefault(resolved[profile], set()).add(group)
    return groups


def get_friends(client: VKClient, user: int) -> list[int]:
    """Получает полный список ID друзей одного профиля с учётом страниц ответа.

    Запрашивает до 5000 записей за раз и увеличивает offset на полученное число.
    При ошибке не возвращает частичный список, чтобы тот не попал в кэш как полный.
    """
    friends = []
    while True:
        result = client.call_api("friends.get", {"user_id": user, "count": 5000, "offset": len(friends)})
        items = [int(item) for item in result["items"]]
        friends.extend(items)
        if len(friends) >= int(result["count"]):
            return friends
        if not items:
            raise RuntimeError(f"Incomplete friends.get pagination for {user}")


def progress(label: str, done: int, total: int, errors: int) -> None:
    """Обновляет строку прогресса прохода: обработанные профили и число ошибок."""
    width = 24
    filled = width * done // max(1, total)
    print(f"\r{label}: [{'#' * filled}{'.' * (width - filled)}] {done}/{total}; errors: {errors}",
          end="\n" if done == total else "", file=sys.stderr, flush=True)


def collect_friends(client: VKClient, seeds: dict[int, set[str]], cache: dict[str, list[int]],
                    cache_path: Path, errors: list[dict], passes: int, max_users: int) -> None:
    """Обходит исходные профили и их друзей, дополняя кэш полными списками.

    Каждый новый список сразу сохраняет в cache_path; записи кэша повторно
    не запрашивает. Два прохода открывают два круга вершин; третий запрашивает
    списки второго круга, но не расширяет выбранную область до третьего.
    Ошибки отдельных профилей добавляет в errors, остальные передаёт в main.
    """
    selected = frontier = set(seeds)
    attempted = set()
    for depth in range(passes):
        discoveries = set()
        users = sorted(frontier)
        label = f"Fetch friends, hop {depth + 1}/{passes}"
        progress(label, 0, len(users), len(errors))
        for index, user in enumerate(users, 1):
            key = str(user)
            if key not in cache:
                if max_users and len(attempted) >= max_users:
                    print("\nReached --max-users; saving the partial graph.", file=sys.stderr)
                    return
                # Лимит считает профили, а не страницы запросов friends.get.
                attempted.add(user)
                try:
                    friends = get_friends(client, user)
                except (VKError, RuntimeError) as exc:
                    errors.append({"user_id": user, "depth": depth, "error": str(exc)})
                    # Ошибки доступа относятся к одному профилю; остальные ошибки останавливают сбор.
                    if not isinstance(exc, VKError) or exc.code not in {15, 18, 30, 113, 200}:
                        raise
                else:
                    cache[key] = friends
                    save_json(cache_path, cache)
            # Дополнительный проход уточняет связи границы, не открывая третий круг.
            if depth < 2:
                discoveries.update(cache.get(key, []))
            progress(label, index, len(users), len(errors))
        frontier = discoveries - selected
        selected = selected | discoveries


def export_graph(directory: Path, seeds: dict[int, set[str]], cache: dict[str, list[int]]) -> tuple[int, int]:
    """Строит по всему кэшу граф двух кругов и записывает nodes.csv и edges.csv.

    Восстанавливает множество вершин в том числе из списков прежних запусков.
    Сохраняет только внутренние связи, без петель и повторов; возвращает
    число записанных вершин и рёбер.
    """
    selected = frontier = set(seeds)
    for _ in range(2):
        frontier = {friend for user in frontier for friend in cache.get(str(user), [])} - selected
        selected = selected | frontier
    # Множества друзей убирают повторы без хранения миллионов отдельных пар рёбер.
    lists = {int(user): set(friends) for user, friends in cache.items() if int(user) in selected}
    nodes_path, edges_path = directory / "nodes.csv", directory / "edges.csv"
    with nodes_path.with_suffix(".csv.tmp").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "is_seed", "seed_groups", "friends_list_fetched"])
        for user in sorted(selected):
            writer.writerow([user, int(user in seeds), ";".join(sorted(seeds.get(user, set()))), int(user in lists)])
    edge_count = 0
    with edges_path.with_suffix(".csv.tmp").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["source", "target"])
        for source in sorted(lists):
            for target in sorted(lists[source]):
                if target not in selected or source == target:
                    continue
                # Взаимную связь пишем один раз; связь из одного списка тоже сохраняем.
                if source < target or source not in lists.get(target, ()):
                    writer.writerow([min(source, target), max(source, target)])
                    edge_count += 1
    nodes_path.with_suffix(".csv.tmp").replace(nodes_path)
    edges_path.with_suffix(".csv.tmp").replace(edges_path)
    return len(selected), edge_count


def main() -> int:
    """Запускает сбор по аргументам командной строки и экспортирует доступный граф.

    Загружает исходные профили и кэш, обрабатывает остановку и пишет журнал ошибок.
    Возвращает код завершения: 0 — проход окончен или достигнут лимит профилей,
    2 — ошибка, 130 — прерывание пользователем.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, default=Path("seeds.csv"), help="CSV: group,profile or one profile per line")
    parser.add_argument("--out", type=Path, default=Path("data"), help="output directory")
    parser.add_argument("--delay", type=float, default=1.0, help="fixed seconds between requests (default: 1)")
    parser.add_argument("--max-users", type=int, default=0, help="limit newly requested profiles; 0 = unlimited")
    parser.add_argument("--complete-boundary", action="store_true", help="also fetch second-hop lists to find edges within the boundary")
    args = parser.parse_args()
    token = os.environ.get("VK_TOKEN")
    if not token:
        parser.error("Set VK_TOKEN in the environment")
    if not math.isfinite(args.delay) or args.delay < 0 or args.max_users < 0:
        parser.error("--delay must be finite and non-negative; --max-users must be non-negative")
    try:
        records = load_seeds(args.seeds)
        if not records:
            raise ValueError("No seed profiles found")
        args.out.mkdir(parents=True, exist_ok=True)
        cache_path = args.out / "friends_cache.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        client = VKClient(token, args.delay)
        print("Resolving seed profiles...", file=sys.stderr, flush=True)
        seeds = resolve_seeds(client, records)
    except KeyboardInterrupt:
        print("\nInterrupted before collection; existing data is unchanged.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Cannot start collection: {exc}", file=sys.stderr)
        return 2

    errors = []
    status = 0
    try:
        collect_friends(client, seeds, cache, cache_path, errors, 3 if args.complete_boundary else 2, args.max_users)
    except KeyboardInterrupt:
        print("\nInterrupted; saving the partial graph.", file=sys.stderr)
        status = 130
    except RuntimeError as exc:
        print(f"\nCollection stopped: {exc}. Saving the partial graph.", file=sys.stderr)
        status = 2
    save_json(cache_path, cache)
    nodes, edges = export_graph(args.out, seeds, cache)
    with (args.out / "errors.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["user_id", "depth", "error"])
        writer.writeheader()
        writer.writerows(errors)
    print(f"Nodes: {nodes:,}; edges: {edges:,}; errors: {len(errors)}; output: {args.out.resolve()}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
