from __future__ import annotations

import logging
import re
import time

from google.cloud import bigquery as bq
from prompt_toolkit.completion import Completer, Completion

log = logging.getLogger(__name__)


class BigQueryCompleter(Completer):
    _TOKEN_RE = re.compile(r"[`A-Za-z0-9_\-\.]+$")

    def __init__(self, repl, cache_ttl_seconds: int = 600) -> None:
        self.repl = repl
        self.cache_ttl_seconds = cache_ttl_seconds
        self._datasets_cache: tuple[float, list[str]] | None = None
        self._tables_cache: dict[str, tuple[float, list[str]]] = {}
        self._columns_cache: dict[tuple[str, str], tuple[float, list[str]]] = {}

    def _now(self) -> float:
        return time.monotonic()

    def _cache_is_fresh(self, cache_entry: tuple[float, list[str]] | None) -> bool:
        if not cache_entry:
            return False
        return (self._now() - cache_entry[0]) < self.cache_ttl_seconds

    def _ensure_client(self) -> bq.Client | None:
        try:
            if not self.repl.client:
                self.repl.client, self.repl.dry_client = self.repl._get_bq_client()
            return self.repl.client
        except Exception as e:
            log.debug("BigQuery client init failed: %s", e)
            return None

    def _get_datasets(self) -> list[str]:
        if self._cache_is_fresh(self._datasets_cache):
            return self._datasets_cache[1]
        client = self._ensure_client()
        if not client:
            return []
        try:
            datasets = [d.dataset_id for d in client.list_datasets(self.repl.PROJECT_ID)]
            datasets.sort()
            self._datasets_cache = (self._now(), datasets)
            return datasets
        except Exception as e:
            log.debug("BigQuery dataset listing failed: %s", e)
            return []

    def _get_tables(self, dataset_id: str) -> list[str]:
        cached = self._tables_cache.get(dataset_id)
        if self._cache_is_fresh(cached):
            return cached[1]
        client = self._ensure_client()
        if not client:
            return []
        try:
            dataset_ref = bq.DatasetReference(self.repl.PROJECT_ID, dataset_id)
            tables = [t.table_id for t in client.list_tables(dataset_ref)]
            tables.sort()
            self._tables_cache[dataset_id] = (self._now(), tables)
            return tables
        except Exception as e:
            log.debug("BigQuery table listing failed for %s: %s", dataset_id, e)
            return []

    def _get_columns(self, dataset_id: str, table_id: str) -> list[str]:
        cache_key = (dataset_id, table_id)
        cached = self._columns_cache.get(cache_key)
        if self._cache_is_fresh(cached):
            return cached[1]
        client = self._ensure_client()
        if not client:
            return []
        try:
            table_ref = f"{self.repl.PROJECT_ID}.{dataset_id}.{table_id}"
            table = client.get_table(table_ref)
            columns = [field.name for field in table.schema]
            columns.sort()
            self._columns_cache[cache_key] = (self._now(), columns)
            return columns
        except Exception as e:
            log.debug("BigQuery schema fetch failed for %s.%s: %s", dataset_id, table_id, e)
            return []

    def _columns_for_table_path(self, table_path: str) -> list[str]:
        raw = table_path.strip("`")
        parts = raw.split(".")
        if len(parts) == 3:
            project_id, dataset_id, table_id = parts
            if project_id != self.repl.PROJECT_ID:
                return []
            return self._get_columns(dataset_id, table_id)
        if len(parts) == 2:
            dataset_id, table_id = parts
            return self._get_columns(dataset_id, table_id)
        if len(parts) == 1:
            return self._get_columns(self.repl.DATASET_ID, parts[0])
        return []

    def _extract_token(self, text: str) -> str:
        match = self._TOKEN_RE.search(text)
        return match.group(0) if match else ""

    def _with_backticks(self, token: str, completion: str) -> str:
        if token.startswith("`"):
            return f"`{completion}`"
        return completion

    def _make_completion(self, token: str, completion: str) -> Completion:
        return Completion(
            self._with_backticks(token, completion),
            start_position=-len(token),
        )

    def get_completions(self, document, complete_event):
        if self.repl.prompt_style != "sql":
            return

        token = self._extract_token(document.text_before_cursor)
        if not token:
            return

        raw = token.strip("`")
        if not raw:
            return

        parts = raw.split(".")
        if not parts:
            return

        current = parts[-1]
        base_parts = parts[:-1]

        # Single segment: suggest datasets (and project if it matches).
        if len(base_parts) == 0:
            if self.repl.PROJECT_ID.startswith(current):
                yield self._make_completion(token, self.repl.PROJECT_ID)
            for dataset in self._get_datasets():
                if dataset.startswith(current):
                    yield self._make_completion(token, dataset)
            return

        # "project." -> datasets; "dataset." -> tables.
        if len(base_parts) == 1:
            first = base_parts[0]
            if first == self.repl.PROJECT_ID:
                for dataset in self._get_datasets():
                    if dataset.startswith(current):
                        yield self._make_completion(token, f"{first}.{dataset}")
            else:
                for table in self._get_tables(first):
                    if table.startswith(current):
                        yield self._make_completion(token, f"{first}.{table}")
            return

        # "project.dataset." -> tables; "dataset.table." -> columns.
        if len(base_parts) == 2:
            first, second = base_parts
            if first == self.repl.PROJECT_ID:
                for table in self._get_tables(second):
                    if table.startswith(current):
                        yield self._make_completion(token, f"{first}.{second}.{table}")
            else:
                for column in self._get_columns(first, second):
                    if column.startswith(current):
                        yield self._make_completion(token, f"{first}.{second}.{column}")
            return

        # "project.dataset.table." -> columns.
        if len(base_parts) == 3 and base_parts[0] == self.repl.PROJECT_ID:
            _, dataset_id, table_id = base_parts
            for column in self._get_columns(dataset_id, table_id):
                if column.startswith(current):
                    yield self._make_completion(
                        token, f"{self.repl.PROJECT_ID}.{dataset_id}.{table_id}.{column}"
                    )
            return


class SQLKeywordCompleter(Completer):
    _TOKEN_RE = re.compile(r"[A-Za-z_]+$")

    def __init__(self, repl, keywords: list[str]) -> None:
        self.repl = repl
        self.keywords = keywords

    def _extract_token(self, text: str) -> str:
        match = self._TOKEN_RE.search(text)
        return match.group(0) if match else ""

    def get_completions(self, document, complete_event):
        if self.repl.prompt_style != "sql":
            return

        token = self._extract_token(document.text_before_cursor)
        if not token:
            return

        token_upper = token.upper()
        for keyword in self.keywords:
            if keyword.startswith(token_upper):
                yield Completion(keyword, start_position=-len(token))


class SQLReplCompleter(Completer):
    def __init__(
        self,
        repl,
        python_completer: Completer,
        sql_completer: BigQueryCompleter,
        keyword_completer: Completer,
    ) -> None:
        self.repl = repl
        self.python_completer = python_completer
        self.sql_completer = sql_completer
        self.keyword_completer = keyword_completer
        self._source_context_re = re.compile(
            r"\b(FROM|JOIN|INTO|TABLE|TABLESAMPLE)\b", re.IGNORECASE
        )
        self._column_context_re = re.compile(
            r"\b(SELECT|WHERE|ON|GROUP\s+BY|HAVING|QUALIFY|ORDER\s+BY)\b",
            re.IGNORECASE,
        )
        self._table_after_source_re = re.compile(
            r"\b(?:FROM|JOIN|INTO|TABLE|TABLESAMPLE)\s+([`A-Za-z0-9_\-\.]+)",
            re.IGNORECASE,
        )
        self._column_token_re = re.compile(r"[A-Za-z0-9_\.]+$")

    def _sql_context(self, document) -> str:
        text = document.text_before_cursor
        if not text:
            return "default"
        last_source = None
        last_column = None
        for last_source in self._source_context_re.finditer(text):
            pass
        for last_column in self._column_context_re.finditer(text):
            pass
        if last_source and (not last_column or last_source.end() > last_column.end()):
            return "sources"
        if last_column:
            return "columns"
        return "default"

    def _last_table_path(self, document) -> str | None:
        text = document.text_before_cursor
        if not text:
            return None
        match = None
        for match in self._table_after_source_re.finditer(text):
            pass
        if match:
            return match.group(1)
        return None

    def _column_token(self, document) -> tuple[str, str]:
        text = document.text_before_cursor
        match = self._column_token_re.search(text)
        token = match.group(0) if match else ""
        if "." in token:
            prefix, suffix = token.rsplit(".", 1)
            return prefix, suffix
        return "", token

    def get_completions(self, document, complete_event):
        if self.repl.prompt_style == "sql":
            context = self._sql_context(document)
            if context == "sources":
                yield from self.sql_completer.get_completions(document, complete_event)
                return
            if context == "columns":
                table_path = self._last_table_path(document)
                if table_path:
                    prefix, token = self._column_token(document)
                    columns = self.sql_completer._columns_for_table_path(table_path)
                    if token:
                        for column in columns:
                            if column.startswith(token):
                                completion = f"{prefix}.{column}" if prefix else column
                                yield Completion(completion, start_position=-len(token))
                        return
                    for column in columns:
                        completion = f"{prefix}.{column}" if prefix else column
                        yield Completion(completion, start_position=0)
                    return
            seen: set[str] = set()
            for completion in self.keyword_completer.get_completions(document, complete_event):
                if completion.text not in seen:
                    seen.add(completion.text)
                    yield completion
            for completion in self.sql_completer.get_completions(document, complete_event):
                if completion.text not in seen:
                    seen.add(completion.text)
                    yield completion
        else:
            yield from self.python_completer.get_completions(document, complete_event)
