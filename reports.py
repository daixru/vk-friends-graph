"""Потоковое чтение центральностей и сохранение компактных рейтингов в CSV и Markdown."""

import csv
import heapq
import json
import math
import sys
from pathlib import Path


def export_top_rankings(source: Path, top: int = 10) -> Path:
    """Создаёт отдельные CSV топов и общий Markdown-отчёт по трём центральностям.

    Читает исходный CSV один раз, удерживая в памяти только top лучших записей
    для каждой метрики. Точность расчёта определяет по метаданным, а не имени
    файла; при равных значениях ставит меньший ID выше. Проверяет конечность
    оценок и согласованность числа строк с метаданными. Возвращает путь к отчёту.
    """
    if top < 1:
        raise ValueError("top must be positive")
    before = source.stat()
    metadata_path = source.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metrics = ("betweenness", "closeness", "eigenvector")
    # Корень каждой кучи — худшая запись текущего топа: её первой заменяет лучший кандидат.
    heaps: dict[str, list[tuple[float, int]]] = {name: [] for name in metrics}
    rows = 0
    with source.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            user = int(row["id"])
            for name in metrics:
                score = float(row[name])
                if not math.isfinite(score):
                    raise ValueError(f"non-finite {name} for user {user}")
                item = (score, -user)  # При равной оценке меньший ID должен стоять выше.
                heap = heaps[name]
                if len(heap) < top:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
            rows += 1
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("centrality CSV changed while exporting rankings")
    if "vertex_count" in metadata and rows != metadata["vertex_count"]:
        raise ValueError("centrality CSV row count does not match its method metadata")
    # Имя файла может вводить в заблуждение; способ расчёта берём только из метаданных.
    methods = {
        "betweenness": ("exact_on_observed_graph" if metadata.get("betweenness_method") == "exact"
                        else "approximate" if "betweenness_method" in metadata else "unknown"),
        "closeness": ("exact_on_observed_graph" if metadata.get("closeness_method") == "exact WF"
                      else "approximate" if "closeness_method" in metadata else "unknown"),
        "eigenvector": "numerical_no_sampling" if "eigenvector_method" in metadata else "unknown",
    }
    titles = {"betweenness": "Посредничество", "closeness": "Близость", "eigenvector": "Собственный вектор"}
    descriptions = {"exact_on_observed_graph": "точный расчёт на наблюдаемом графе",
                    "approximate": "приближённая оценка", "numerical_no_sampling": "численный расчёт без выборки",
                    "unknown": "метод неизвестен: отсутствуют метаданные"}
    report = [f"# Топ-{top} по центральности", "", f"Рейтинг среди **{rows:,} профилей** в текущем результате.",
              "Все показатели относятся к собранному графу; отсутствующие связи могут изменить рейтинг.",
              "При одинаковом значении порядок определяется числовым ID. Топ не пересчитывает сами показатели.", ""]
    if metadata.get("samples"):
        report += [f"Размер выборки: {metadata['samples']}; random seed: {metadata.get('random_seed', 'unknown')}.", ""]
    for name in metrics:
        ranked = sorted(heaps[name], reverse=True)
        path = source.with_name(f"{source.stem}_top{top}_{name}.csv")
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["rank", "id", "profile_url", "value", "method"])
            for rank, (score, negative_user) in enumerate(ranked, 1):
                user = -negative_user
                writer.writerow([rank, user, f"https://vk.ru/id{user}", score, methods[name]])
        temporary.replace(path)
        report += [f"## {titles[name]}", "", f"Метод: **{descriptions[methods[name]]}**.", ""]
        if name == "eigenvector" and metadata.get("component_count", 1) > 1:
            report += ["Нормировка выполнена отдельно по компонентам: значения из разных компонент несопоставимы.", ""]
        report += ["| Место | Профиль VK | Значение |", "| --- | --- | ---: |"]
        for rank, (score, negative_user) in enumerate(ranked, 1):
            user = -negative_user
            report.append(f"| {rank} | [id{user}](https://vk.ru/id{user}) | {score:.10g} |")
        report += [""]
        print(f"Top {top} {name}: {path} ({methods[name]})", file=sys.stderr)
    report_path = source.with_name(f"{source.stem}_top{top}.md")
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(report), encoding="utf-8")
    temporary.replace(report_path)
    return report_path


