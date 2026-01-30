import os
import site
from jupyter_client.blocking.client import BlockingKernelClient
from site import ENABLE_USER_SITE
from pathlib import Path
import sys
import asyncio
import re
import time
from datetime import datetime, date
import contextlib
import logging
from textwrap import dedent
from tqdm import tqdm
import db_dtypes
from prompt_toolkit.filters import vi_navigation_mode, Condition
from prompt_toolkit.key_binding.vi_state import InputMode
from pygments.lexers.sql import GoogleSqlLexer
from pygments.lexers.python import PythonLexer
import pyarrow as pa
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.formatted_text import HTML, AnyFormattedText
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit import print_formatted_text

# from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard
from sqlrepl.osc_clipboard import OSCClipboard
import ptpython
from ptpython.prompt_style import PromptStyle
from ptpython.repl import PythonRepl
from ptpython.python_input import PythonInput

# from ptpython.ipython import InteractiveShellEmbed
from sqlrepl.status_bar import status_bar
from sqlrepl.style import mystyle
import click
from zoneinfo import ZoneInfo
from rich import print, get_console, inspect as ins

insm = lambda x: ins(x, methods=True)
insa = lambda x: ins(x, all=True)
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.errors import NotRenderableError
from rich.pretty import pprint
from rich.logging import RichHandler
from rich.progress import track

# from rich.markdown import Markdown
import google.cloud.bigquery as bq
import google.cloud.bigquery_storage as bqs
from google.api_core.exceptions import GoogleAPICallError

log = logging.getLogger()


is_linux = os.getenv("IS_LINUX") == "true"


import pandas as pd
import numpy as np

pd.options.display.max_rows = 4000
pd.options.display.float_format = "{:.2f}".format

utcpd = pd.DatetimeTZDtype(tz="UTC")

pandas_bq_types = dict(
    bool_dtype=pa.bool_(),
    int_dtype=pa.int64(),
    float_dtype=pa.float64(),
    string_dtype=pa.string(),
    date_dtype=pa.timestamp("us", tz="UTC"),
    datetime_dtype=pa.timestamp("us", tz="UTC"),
    timestamp_dtype=pa.timestamp("us", tz="UTC"),
)

pandas_bq_types = {k: pd.ArrowDtype(v) for k, v in pandas_bq_types.items()}


bqtypes = {
    "BOOL": pandas_bq_types["bool_dtype"],
    "INTEGER": pandas_bq_types["int_dtype"],
    "FLOAT": pandas_bq_types["float_dtype"],
    "STRING": pandas_bq_types["string_dtype"],
    "DATE": pandas_bq_types["date_dtype"],
    "DATETIME": pandas_bq_types["date_dtype"],
    "TIMESTAMP": pandas_bq_types["date_dtype"],
}


from decimal import Decimal
import sqlfluff


try:
    jinja_params = dict(
        ENV="dev",
        PROJECT_ID=os.environ["PROJECT_ID"],
        DATASET_ID=os.environ["PROJECT_ID"] + "." + os.environ["DATASET_ID"],
        DEC_DATASET_ID=os.environ["PROJECT_ID"] + "." + os.environ["DEC_DATASET_ID"],
        VOLTAGE_DATASET=os.environ["PROJECT_ID"] + "." + os.environ["VOLTAGE_DATASET"],
    )
except KeyError as e:
    print(
        "Must set PROJECT_ID, DATASET_ID, DEC_DATASET_ID, & VOLTAGE_DATASET environment variables to start SQLREPL"
    )
    sys.exit(1)


sqlkeywords = [
    "ALL",
    "AND",
    "ANY",
    "ARRAY",
    "AS",
    "ASC",
    "ASSERT_ROWS_MODIFIED",
    "AT",
    "BETWEEN",
    "BY",
    "CASE",
    "CAST",
    "COLLATE",
    "CONTAINS",
    "CREATE",
    "CROSS",
    "CUBE",
    "CURRENT",
    "DEFAULT",
    "DEFINE",
    "DESC",
    "DISTINCT",
    "ELSE",
    "END",
    "ENUM",
    "ESCAPE",
    "EXCEPT",
    "EXCLUDE",
    "EXISTS",
    "EXTRACT",
    "FALSE",
    "FETCH",
    "FOLLOWING",
    "FOR",
    "FROM",
    "FULL",
    "GROUP",
    "GROUPING",
    "GROUPS",
    "HASH",
    "HAVING",
    "IF",
    "IGNORE",
    "IN",
    "INNER",
    "INTERSECT",
    "INTERVAL",
    "INTO",
    "IS",
    "JOIN",
    "LATERAL",
    "LEFT",
    "LIKE",
    "LIMIT",
    "LOOKUP",
    "MERGE",
    "NATURAL",
    "NEW",
    "NO",
    "NOT",
    "NULL",
    "NULLS",
    "OF",
    "ON",
    "OR",
    "ORDER",
    "OUTER",
    "OVER",
    "PARTITION",
    "PRECEDING",
    "PROTO",
    "QUALIFY",
    "RANGE",
    "RECURSIVE",
    "RESPECT",
    "RIGHT",
    "ROLLUP",
    "ROWS",
    "SELECT",
    "SET",
    "SOME",
    "STRUCT",
    "TABLESAMPLE",
    "THEN",
    "TO",
    "TREAT",
    "TRUE",
    "UNBOUNDED",
    "UNION",
    "UNNEST",
    "USING",
    "WHEN",
    "WHERE",
    "WINDOW",
    "WITH",
    "WITHIN",
]


def format_fix(query):
    for key, value in jinja_params.items():
        query = query.replace("{{" + key + "}}", value)
    return sqlfluff.fix(query, config_path=os.environ["HOME"] + "/.sqlfluff")


# pretty.install()
eastern = ZoneInfo("US/Eastern")

os.environ["MANPAGER"] = "bat --language=py -p"
os.environ["PAGER"] = "bat --language=py -p"


def showhelp():
    get_console().print(
        """
Statements starting with a capitalized SQL keyword are executed as [blue]SQL[/], others as [green]Python[/].
All [blue]SQL[/] statements are automatically formatted before execution and copied to your clipboard if they succeed.
\n
Keys:
[magenta]F8[/]: switch between [blue]SQL[/] and [green]Python[/] modes (or just type [blue]sql[/] or [green]python[/] and press enter)
[magenta]Ctrl-D[/]: exit
[magenta]Ctrl-Q[/]: exit
\n
Global Variables:
[green]df[/]: last query result as pandas DataFrame
[green]dfs[/]: list of past query results as pandas DataFrames
\n
Commands:
[blue]lookup[/] <tablename>: look up a BigQuery table's schema and details
    for example: [yellow]lookup [blue]clin_analytics_hcb_dev.cm_case_prep_common_membership[/][/]
[blue]bqsession[/]: start a BigQuery session with persistent temp tables and declarations
[blue]bqendsession[/]: end the current BigQuery session
[blue]checkrunning[/]: check for currently running BigQuery jobs and refresh these globals
    [green]running_jobs[/]: list of currently running BigQuery jobs
    [green]bqjobs[/]: list of last 20 BigQuery jobs
\n
Useful Functions:
[magenta]help[/]([yellow]something[/]) to read a [green]__doc__[/]
[magenta]ins[/]([yellow]object[/]) to inspect an object
\n
[red]IMPORTANT:[/] Queries are submitted asynchronousy. [magenta]Ctrl-C[/] stops waiting for results but does not cancel the query. To cancel the query, run [yellow]checkrunning[/], then running_jobs[0].cancel().
Before exiting, the REPL will attempt cancel of all of your running BQ jobs, if there are any.
"""
    )


def help(someobj: object) -> str | None:
    # docstring = someobj.__doc__
    # if not docstring:
    #     return f"No docstring for {someobj}"
    #
    # # Regular expression to find "Parameters" and parameter names at the start of a line followed by a colon
    # param_pattern = re.compile(r"(?<=\n)[ \t]*(\s*)(\w+)(\s*:\s*)(.*)", re.MULTILINE)
    #
    # # Function to apply blue color to parameter names and "Parameters"
    # def apply_blue(match):
    #     space = match.group(1) if match.group(1) else ''
    #     word = match.group(2)
    #     rest = match.group(4) if match.group(4) else ''
    #     return f"{space}[blue]{word}[/]: [yellow]{rest}[/]"
    #
    # # Apply the regex substitution
    # result = param_pattern.sub(apply_blue, docstring)
    # result = result.replace('Parameters','[blue]Parameters[/]')

    c = get_console()
    with c.pager():
        c.print(someobj.__doc__, highlight=False, markup=True)


#     return result


def printdf(dataframe, title="Dataframe", color_by="dtype") -> None:
    """Display dataframe as table using rich library.
    Args:
        df (pd.DataFrame): dataframe to display
        title (str, optional): title of the table. Defaults to "Dataframe".
    Raises:
        NotRenderableError: if dataframe cannot be rendered
    Returns:
        rich.table.Table: rich table
    """

    colors = {
        "object": "blue",
        "float64": "green",
        "int64": "magenta",
        "bool": "cyan",
        "datetime64[ns]": "yellow",
    }

    rowcolor = False
    if color_by != "dtype":
        colors = {
            str: "blue",
            float: "green",
            int: "magenta",
            bool: "cyan",
            date: "yellow",
            datetime: "yellow",
            pd.Timestamp: "yellow",
            Decimal: "green",
        }
        rowcolor = True

    table = Table(title=title)
    dataframe = dataframe.copy()
    color = ""
    for col, dtype in dataframe.dtypes.items():
        if color_by == "dtype":
            color = colors.get(str(dtype), "")
        table.add_column(col, header_style=f"bold {color}", style=color)

    for idx, row in dataframe.iterrows():
        if rowcolor:
            rowtype = str(type(row[color_by])).lower()
            color = colors.get(rowtype)

        with contextlib.suppress(NotRenderableError):
            table.add_row(*row.astype(str), style=color)

    get_console().print(table)


class MyPrompt(PromptStyle):

    def __init__(self, python_input: PythonInput, prompt_title: str) -> None:
        self.python_input = python_input
        self.prompt_title = prompt_title
        colors = {
            "SQL": {"fg": "ansiblue"},
            "Py": {"fg": "ansigreen", "idx": "ansiyellow"},
            "iPy": {"fg": "#289c36", "idx": "#2080D0"},
            "Debug": {"fg": "ansired"},
            "Async": {"fg": "ansired"},
            "DebugAsync": {"fg": "ansired"},
        }
        main_color = colors.get(prompt_title, {})
        self.color = main_color.get("fg", "ansiblue")
        self.idx_color = main_color.get("idx")

    def in_prompt(self) -> AnyFormattedText:
        idx = self.python_input.current_statement_index
        if self.idx_color:
            idx = f"<style fg='{self.idx_color}'>{idx}</style>"
        title = self.prompt_title
        return HTML(f"<style fg='{self.color}'>{title} [{idx}]</style>: ")

    def in2_prompt(self, width: int) -> AnyFormattedText:
        return "...: ".rjust(width)

    def out_prompt(self) -> AnyFormattedText:
        idx = self.python_input.current_statement_index
        color = "ansired"
        return HTML(f"<{color}>Result [{idx}]</{color}>: ")


# BigQuery completion support for SQL prompt.
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
        sql_completer: "BigQueryCompleter",
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


# def MyValidator(Validator):
#
#     @abstractmethod
#     def validate(self, document: Document) -> None:
#         if


class MyRpl(PythonRepl):

    # def _ensure_nvim(self) -> bool:
    #     print = self.c.print
    #     if not os.path.exists(self.NVIM_LISTEN_ADDRESS):
    #         return False
    #
    #     new = ""
    #     if self.nvim:
    #         try:
    #             self.nvim.current.buffer
    #             return True
    #         except:
    #             self.nvim = None
    #             new = "new"
    #
    #     if not self.nvim:
    #         try:
    #             self.nvim = attach("socket", path=self.NVIM_LISTEN_ADDRESS)
    #             tmux_pane = os.getenv("TMUX_PANE")
    #             print(f"Attached to {new} NVIM")
    #             if tmux_pane:
    #                 os.system(f"tmux setenv -g TMUX_TARGET_ID ''{tmux_pane}''")
    #                 self.nvim.api.set_var("tmux_target_id", tmux_pane)
    #             return True
    #         except Exception as e:
    #             print("Error attaching to nvim:", e)
    #
    #     return False

    def init_console(self):
        """
        Initialize rich console, which will be the same tty the repl is running in,
        unless the following conditions are met:
        1. pipe_logs is True
        2. running in tmux
        3. there are exactly 3 panes in the tmux session
        In which case the console will be redirected to the second pane.

        """

        self.c = Console(color_system="truecolor")
        if not self.pipe_logs or not os.getenv("TMUX_PANE"):
            return

        allpanes = os.popen('tmux list-panes -F "#{pane_id} #{pane_tty}"').readlines()
        if len(allpanes) != 3:
            return

        log_to_id, log_to_tty = allpanes[1].strip().split()
        for h in log.handlers:
            if isinstance(h, RichHandler):
                self.c.file = open(log_to_tty, "w")
                h.console = self.c
                h.console.clear()
                log.info(f"Logging to tmux pane {log_to_id}")

    def __init__(
        self,
        title: str | None = None,
        debug_mode: bool = False,
        is_async: bool = False,
        pipe_logs: bool = False,
        *args,
        **kwargs,
    ) -> None:
        kwargs["vi_mode"] = os.getenv("SQLREPL_VI")
        kwargs["history_filename"] = os.environ["HOME"] + "/ptpython_history_sql"
        kwargs["_extra_toolbars"] = [status_bar(self)]
        super().__init__(*args, **kwargs)
        self._python_completer = self.completer
        self._bq_completer = BigQueryCompleter(self)
        self._keyword_completer = SQLKeywordCompleter(self, sqlkeywords)
        self.completer = SQLReplCompleter(
            repl=self,
            python_completer=self._python_completer,
            sql_completer=self._bq_completer,
            keyword_completer=self._keyword_completer,
        )
        self.debug_mode = debug_mode
        self.async_loop = is_async
        self.pipe_logs = pipe_logs
        self.bq_storage_mode = os.getenv("SQLREPL_USE_BQ_STORAGE", False)
        self.async_debug = self.debug_mode and self.async_loop
        self.storage_client = None
        self.title = title or "SqlRepl"
        self.show_custom_status_bar = title
        self.style = "dracula"
        if os.environ.get("OS_THEME") == "light":
            self.style = "default"
        self.use_code_colorscheme(self.style)

        self.ui_styles = {"default": mystyle}
        self.use_ui_colorscheme("default")

        self._lexer = PygmentsLexer(PythonLexer)
        # self._extra_toolbars = [status_bar(self)]
        # self.ptpython_layout = PtPythonLayout(
        # self,
        # lexer=DynamicLexer(
        # lambda: self._lexer
        # if self.enable_syntax_highlighting
        # else SimpleLexer()
        # ),
        # input_buffer_height=self._input_buffer_height,
        # extra_buffer_processors=self._extra_buffer_processors,
        # extra_body=self._extra_layout_body,
        # extra_toolbars=self._extra_toolbars,
        # )
        self.all_prompt_styles["sql"] = MyPrompt(self, "SQL")
        self.all_prompt_styles["python"] = MyPrompt(self, "Py")
        self.all_prompt_styles["ipython"] = MyPrompt(self, "iPy")
        if self.debug_mode:
            self.all_prompt_styles["python"] = MyPrompt(self, "Debug")
        if self.async_loop:
            self.all_prompt_styles["python"] = MyPrompt(self, "Async")
        if self.async_debug:
            self.all_prompt_styles["python"] = MyPrompt(self, "DebugAsync")
        self.prompt_style = "python"
        self.confirm_exit = True
        self.enable_input_validation = False
        # self._validator = Validator.from_callable(func=lambda text: bool(text.strip()), error_message="Input cannot be empty", move_cursor_to_end=True)
        self.show_docstring = True
        self.show_status_bar = False
        self.ptpython_layout.status_bar = status_bar
        self.terminal_title = "OMREPL"
        self.enable_open_in_editor = False
        self.enable_auto_suggest = True
        self.enable_history_search = True
        self.enable_output_formatting = True
        self.enable_dictionary_completion = True
        self.enable_pager = True
        self.complete_while_typing = False
        self.wrap_lines = True
        self.show_line_numbers = True
        self.highlight_matching_parenthesis = True
        self.init_console()
        self.vi_start_in_navigation_mode = True
        self.vi_keep_last_used_mode = True
        self.insert_blank_line_after_output = True
        self.cursor_shape_config = "Modal (vi)"
        self.nvim = None
        self.dfs = []
        self.client = None
        self.PROJECT_ID = os.environ["PROJECT_ID"]
        self.DATASET_ID = os.environ["DATASET_ID"]
        # self.NVIM_LISTEN_ADDRESS = os.environ["NVIM_SOCK"]
        # self._ensure_nvim()  # Initialize nvim context
        # self.c = Console()
        # self.get_globals = self.get_globals
        # self.get_locals = self.get_locals

        self.ipython_mode = False
        if self.ipython_mode:
            cf = "/Users/n856925/Library/Jupyter/runtime/test.json"
            self.kc = BlockingKernelClient(connection_file=cf)
            self.kc.load_connection_file()
            self.kc.start_channels()
            self.prompt_style = "ipython"

    def _get_bq_client(self) -> tuple[bq.Client, bq.Client]:

        print("[blue]Connecting...", end=" ")

        # Instantiating client does this step anyway so it doesnt add any extra time
        from google.auth import default

        credentials, _ = default()

        default_dataset = bq.DatasetReference(self.PROJECT_ID, self.DATASET_ID)
        default_config = bq.QueryJobConfig(
            default_dataset=default_dataset, job_timeout_ms=600000  # 10 min timeout
        )
        dry_run_config = bq.QueryJobConfig(
            default_dataset=default_dataset, dry_run=True, use_query_cache=False
        )
        client = bq.Client(
            credentials=credentials,
            default_query_job_config=default_config,
        )
        dry_client = bq.Client(
            project=self.PROJECT_ID,
            credentials=credentials,
            default_query_job_config=dry_run_config,
        )
        print("[green]Done", end="\r")

        if self.bq_storage_mode:
            self.storage_client = bqs.BigQueryReadClient(credentials=credentials)

        return client, dry_client

    def do_ipython(self, line: str):
        self.kc.execute(f"_console.width = {self.c.width}")
        msg_id = self.kc.execute(line)
        result = None
        while True:
            msg = self.kc.get_iopub_msg()
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            mtype = msg["header"]["msg_type"]
            content = msg["content"]
            # print(f"DEBUG: mtype={mtype}, content={content}")
            if mtype == "stream":  # print/output
                print(content["text"], end="")
            elif mtype in ("display_data", "execute_result"):  # input prompt
                result = content["data"].get("text/plain", "")
            elif mtype == "error":  # rich traceback pieces
                result = "\n".join(content["traceback"])
            elif mtype == "status" and content["execution_state"] == "idle":
                break

        if result:
            print_formatted_text(ANSI(result))

    def do_python(self, line: str):

        globals = self.get_globals()
        try:
            # if self._ensure_nvim():
            # This causes an issue with dask distributed unless it's protected by __name__==__main__
            # globals["__file__"] = self.nvim.current.buffer.name
            output = super().eval(line)
            return output
        except Exception as e:
            globals["last_exception"] = e
            # globals["last_frame"] = sys
            get_console().print_exception(show_locals=False, suppress=[ptpython], max_frames=10)

    def do_sql(self, query: str) -> object:
        if not self.client:
            self.client, self.dry_client = self._get_bq_client()

        globals = self.get_globals()
        try:
            query_job = self.dry_client.query(query)
            bytes_billed = round((query_job.total_bytes_processed or 0) / 1e9, 3)
            estmsg = f"Est {bytes_billed} GB"
            self.c.print(f"[dim]{estmsg}", end="... ")
        except Exception as e:
            print(f"\n{e}")
            return
        print("[green]Submitted[/]", end="\r")
        is_select = query.startswith(("SELECT", "WITH"))
        query = f'SET @@dataset_project_id="{self.PROJECT_ID}";\n{query}'
        query_job = self.client.query(query)

        # This is only necessary because failed assertions are not caught by the dry run
        # and will raise an exception when the query is run. (A 400 Bad Request for some reason)
        try:
            res = query_job.result()
        except GoogleAPICallError as e:
            if query_job.statement_type == "ASSERT":
                print("\n[red]BIGQUERY ASSERTION ERROR:")
                print(f"{e.errors[0]['message']}\n")
                return
            print(f"\nError: {e}\n")
            return

        self.app.clipboard.set_text(query)
        color = "dim"
        bytes_billed = round((query_job.total_bytes_billed or 0) / 1e9, 3)
        if bytes_billed:
            color = "red" if bytes_billed >= 500 else "yellow" if bytes_billed >= 100 else "blue"
        print(f"[dim]{estmsg} -> Done ([{color}]{bytes_billed}[/] GB)")
        # client.query_and_wait is also an option
        if is_select and res.total_rows:
            large_result = res.total_rows > 40000
            if self.bq_storage_mode or (is_linux and large_result):
                df = res.to_dataframe(bqstorage_client=self.storage_client, **pandas_bq_types)  # type: ignore
            else:
                globals["rows"] = []
                globals["dictrows"] = []
                for row in track(
                    res,
                    total=res.total_rows,
                    description=f"Getting {res.total_rows} results...",
                    disable=not large_result,
                    transient=True,
                    console=self.c,
                ):
                    globals["rows"].append(row)
                    globals["dictrows"].append(dict(row))
                df = pd.DataFrame(globals["dictrows"])
                print()

            df.attrs["query"] = query
            df.attrs["meta"] = {k: str(v) for k, v in df.dtypes.items()}

            rows, cols = df.shape
            if rows == 1:
                print(df.T)
            elif cols > 10:
                _max_rows = pd.options.display.max_rows
                pd.options.display.max_rows = 10
                print(df)
                pd.options.display.max_rows = _max_rows
            else:
                print("\n")
                printdf(df.head(20))
                print("\n")
            # globals["df"] = self.d.data = df
            globals["df"] = df
            self.dfs.append(df)
            globals["dfs"] = self.dfs
            return f"Returned {rows} rows, {cols} cols"
        elif is_select:
            return "[bold][green]Query returned zero rows[/]"
        else:
            return f"[bold][green]Finished[/]"

    def _checkrunning(self):
        globals = self.get_globals()
        if not self.client:
            self.client, self.dry_client = self._get_bq_client()
        globals["bqjobs"] = self.bqjobs = list(self.client.list_jobs(max_results=20))
        globals["running_jobs"] = self.running_jobs = list(
            self.client.list_jobs(state_filter="running")
        )
        if self.running_jobs:
            return "There are running jobs. Check `running_jobs`"

    def handle_choice(self, filetype):

        if filetype not in ["sql", "python","ipython"]:
            return

        if filetype == self.prompt_style:
            return

        if filetype == "sql":
            self.prompt_style = "sql"
            self._lexer = PygmentsLexer(GoogleSqlLexer)

        if filetype in ["python","ipython"]:
            self.prompt_style = filetype
            self._lexer = PygmentsLexer(PythonLexer)

        return f"Mode: {self.prompt_style.upper()}"

    def lookup(self, tablename):
        if not self.client:
            self.client, self.dry_client = self._get_bq_client()

        print = self.c.print

        splittable = tablename.split(".")
        if len(splittable) == 1:
            tablename = self.PROJECT_ID + "." + self.DATASET_ID + "." + tablename
        if len(splittable) == 2:
            tablename = self.PROJECT_ID + "." + tablename

        for key, value in jinja_params.items():
            tablename = tablename.replace("{{" + key + "}}", value)

        t = self.client.get_table(tablename)
        print(f"\n{t.reference}\n")
        print(f"Type: {t.table_type}")
        if viewquery := t.view_query:
            try:
                viewquery = (
                    format_fix(t.view_query) if not "insight" in t.dataset_id else t.view_query
                )
            except KeyboardInterrupt:
                print("\n[red]View query display cancelled\n")
            highlighted = Syntax(viewquery, "googlesql", theme=self.style, line_numbers=True)
            print("\n", highlighted, "\n")
        else:
            print(f"{t.num_rows} rows")
        print("Columns")
        colors = {
            "NUMERIC": "yellow",
            "FLOAT": "yellow",
            "INTEGER": "yellow",
            "STRING": "green",
            # "BOOLEAN": "magenta",
            "TIMESTAMP": "cyan",
            "DATE": "cyan",
        }
        for col in t.schema:
            color = colors.get(col.field_type, "reset")
            print(f"[{color}]{col.name} {col.field_type}")

        print("\n")

        if t.modified:
            modified = t.modified.astimezone(eastern).strftime("%Y-%m-%d %I:%M:%S %p ET")
            print(f"Modified at: {modified}")
        if t.created:
            created = t.created.astimezone(eastern).strftime("%Y-%m-%d %I:%M:%S %p ET")
            print(f"Created at: {created}")

        globals = self.get_globals()
        globals["t"] = t
        print("\nTable object is globally assigned to `t` for exploration\n")

    def _accept_handler(self, buff):
        words = buff.text.strip().split()
        if words and words[0] in sqlkeywords:
            self.handle_choice("sql")
            buff.text = format_fix(buff.text)
        elif self.prompt_style == "sql":
            self.handle_choice("python")
        return super()._accept_handler(buff)

    async def eval_async(self, line: str) -> object:
        try:
            x = await super().eval_async(line)
            sys.stdout.flush()
            if x is not None:
                pprint(x, max_length=100)
        except Exception as e:
            log.exception(e)

    def eval(self, line: str) -> object:

        choice = self.handle_choice(line)
        if choice:
            return choice

        if line == "checkrunning":
            return self._checkrunning()

        if line == "reset console":
            self.init_console()
            return "Reset console"

        if line == "bqsession":
            if not self.client:
                self.client, self.dry_client = self._get_bq_client()
            self.client.default_query_job_config.create_session = True
            session_id = self.client.query("SELECT 1").session_info.session_id
            self.client.default_query_job_config.connection_properties = [
                bq.ConnectionProperty("session_id", session_id)
            ]
            self.dry_client.default_query_job_config.connection_properties = [
                bq.ConnectionProperty("session_id", session_id)
            ]
            self.client.default_query_job_config.create_session = False
            self.c.print(f"\n[blue]BigQuery session started")
            self.handle_choice("sql")
            return

        if line == "bqendsession":
            if not self.client:
                return "No active BigQuery client"
            self.client.query("CALL BQ.ABORT_SESSION();").result()
            self.client.default_query_job_config.connection_properties = []
            self.dry_client.default_query_job_config.connection_properties = []
            self.c.print("[blue]Ended BigQuery session")

        if line == "reset_nvim_tries":
            tmux_pane = os.getenv("TMUX_PANE")
            if tmux_pane:
                os.system(f"tmux setenv -g TMUX_TARGET_ID ''{tmux_pane}''")
            return "Reset nvim tries"

        if line == "help":
            showhelp()
            return

        if line.startswith("lookup "):
            self.lookup(line.split()[-1].replace("`", ""))
            return

        dofunc = {
            "sql": self.do_sql,
            "python": self.do_python,
            "ipython": self.do_ipython,
        }
        output = dofunc[self.prompt_style](line)
        has_output = output is not None
        # The best strategy would be to learn how self.style_transformations works in conjunction
        # with the parent class output printer.
        if has_output:
            if isinstance(output, Exception):
                # raise output
                try:
                    raise output
                except Exception as e:
                    self.c.print_exception(
                        show_locals=False,
                        suppress=[ptpython],
                        max_frames=10,
                        extra_lines=10,
                    )
                return
            elif isinstance(output, (list, dict, set)):
                pprint(output, max_length=100)
                return
            self.c.print(output)
            # return output

        # if isinstance(output, pd.DataFrame):
        # print(output)
        # return
        # else:
        # return output

    def _remove_from_namespace(self):
        # Idk what this is supposed to be for but it causes issues
        pass


# The globals and locals have to be set up exactly this way. Exactly nested.
# Otherwise it doesn't work.


def embed(
    debug_mode=False,
    title=None,
    parent_globals=None,
    parent_locals=None,
    return_asyncio_coroutine=False,
    pipe_logs=False,
):

    myglobals = parent_globals or globals()
    mylocals = parent_locals or locals()

    if not debug_mode:
        myglobals["__name__"] = "__main__"

    def get_globals():
        return myglobals

    def get_locals():
        return mylocals

    # Create REPL.
    repl = MyRpl(
        title=title,
        debug_mode=debug_mode,
        get_globals=get_globals,
        get_locals=get_locals,
        is_async=return_asyncio_coroutine,
        pipe_logs=pipe_logs,
    )

    repl.app.clipboard = OSCClipboard()

    @repl.add_key_binding("E", filter=vi_navigation_mode)
    def _(event) -> None:
        event.current_buffer.cursor_position += (
            event.current_buffer.document.get_end_of_line_position()
        )

    @repl.add_key_binding("f6")
    def _(event) -> None:
        repl.ipython_mode = not repl.ipython_mode
        if repl.ipython_mode:
            repl.handle_choice("ipython")
        else:
            repl.handle_choice("python")

    @repl.add_key_binding("f7")
    def _(event) -> None:
        breakpoint()

    @repl.add_key_binding("f8")
    def _(event) -> None:
        if repl.prompt_style == "python":
            repl.handle_choice("sql")
        else:
            repl.handle_choice("python")

    # To debug events do this ( if in the repl, do get_globals = globals )
    # globals = get_globals()
    # globals['event'] = event
    # Then the event can be explored in the repl
    # Slight modification of the default ctrl-c so that if there's no text it doesn't do anything
    @repl.add_key_binding("c-c")
    def _(event) -> None:
        if event.app.current_buffer.text:
            event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    # overwrite this because I use it but never intend to cut text, and it's slow in ssh if xserver is connected
    @repl.add_key_binding("x", filter=vi_navigation_mode)
    def _(event) -> None:
        event.current_buffer.delete()

    # can also use "is_multiline" condition to make ] and [ work when not in multiline
    # https://github.com/prompt-toolkit/ptpython/blob/5021832f76309755097b744f274c4e687a690b85/ptpython/key_bindings.py
    @repl.add_key_binding("c-y")
    def _(event) -> None:
        os.system("tmux copy-mode")

    @repl.add_key_binding("c-u")
    def _(event) -> None:
        os.system("tmux copy-mode")

    def real_exit(event):
        if repl.client:  # If there's a client, check for running jobs
            try:
                repl._checkrunning()
                job_cnt = len(repl.running_jobs)
                repl.c.print(f"\n\n[red]Cancelling [green]{job_cnt}[/] running [blue]BQ[/] jobs")
                for job in repl.running_jobs:
                    repl.client.cancel_job(job.job_id)
            except Exception as e:
                log.error(f"Error cancelling jobs: {e}")
                # log.exception(e)
        event.app.exit(exception=EOFError, style="class:exiting")

    confirmation_visible = Condition(lambda: repl.show_exit_confirmation)

    @repl.add_key_binding("y", filter=confirmation_visible)
    @repl.add_key_binding("Y", filter=confirmation_visible)
    @repl.add_key_binding("enter", filter=confirmation_visible)
    @repl.add_key_binding("c-d", filter=confirmation_visible)
    def _(event) -> None:
        real_exit(event)

    # press gf to go to the file under cursor
    # '/Users/n856925/Documents/github/ptpython/ptpython/key_bindings.py'
    @repl.add_key_binding("c-q", filter=vi_navigation_mode)
    @repl.add_key_binding("c-d", filter=vi_navigation_mode)
    def _(event) -> None:
        if repl.confirm_exit:
            # Show exit confirmation and focus it (focusing is important for
            # making sure the default buffer key bindings are not active).
            repl.show_exit_confirmation = True
            repl.app.layout.focus(repl.ptpython_layout.exit_confirmation)
        else:
            real_exit(event)

    @repl.add_key_binding("B", filter=vi_navigation_mode)
    def _(event) -> None:
        b = event.current_buffer
        b.cursor_position += b.document.get_start_of_line_position(after_whitespace=True)

    @repl.add_key_binding("c-space")
    def _(event) -> None:
        """
        Accept suggestion.
        """
        b = event.current_buffer
        suggestion = b.suggestion

        if suggestion:
            b.insert_text(suggestion.text)

    if repl.vi_mode and repl.vi_start_in_navigation_mode:
        repl.app.vi_state.input_mode = InputMode.NAVIGATION

    if return_asyncio_coroutine:

        async def coroutine() -> None:
            await repl.run_async()

        log.debug("Returning asyncio coroutine")
        return coroutine()  # type: ignore
    else:
        repl.run()
        return None


async def run_asyncio_coroutine(coro):
    """
    Run an asyncio coroutine in the event loop.
    """
    await coro


def export_df(df, name, row_limit=2000):
    if df.index.name:
        df = df.reset_index().reset_index()
    row_limit = row_limit or len(df)
    df.head(row_limit).to_parquet(f"{os.environ['HOME']}/tmp/parq/{name}.parquet", index=False)


top_globals = globals()
top_locals = locals()


def check_project_status():
    paths = {"not a git repo": Path(".git"), "not a pyproject": Path("pyproject.toml")}
    warnlist = [message for message, path in paths.items() if not path.exists()]
    return f"[yellow]({', '.join(warnlist)})" if warnlist else ""


@click.command()
@click.option("--run-async", is_flag=True, help="Async")
@click.option("--verbose", is_flag=True, help="INFO output")
@click.option(
    "--pipe-logs",
    is_flag=True,
    help="pipe logs to tmux pane in top right corner if possible",
)
def cli(run_async, verbose, pipe_logs):
    """Command-line interface for the embed function."""

    # stdout bc otherwise there's softwrap
    sitepackages = site.getsitepackages()[0]
    sitecustomize = os.path.join(sitepackages, "sitecustomize.py")
    has_customize = "[red]not " if not os.path.isfile(sitecustomize) else "[green]"
    sys.path.insert(0, sitepackages)  # ensure it's on top, not bottom

    getreplcmd = "which sqlrepl"
    repl_executable = os.popen(getreplcmd).read().strip()

    cwd = disp_cwd = os.getcwd()
    homedir = os.path.expanduser("~")
    sys_executable = sys.executable
    if os.path.exists(homedir):
        disp_cwd = cwd.replace(homedir, "~")
        sitepackages = sitepackages.replace(homedir, "~")
        repl_executable = repl_executable.replace(homedir, "~")
        sys_executable = sys.executable.replace(homedir, "~")
    # rel_sitepackages = os.path.relpath(sitepackages, cwd)
    # rel_repl = os.path.relpath(repl_executable, cwd)
    # rel_executable = os.path.relpath(sys.executable, cwd)
    warns = check_project_status()

    c = get_console()

    c.rule()
    c.print("[yellow]Welcome to [green]Python-[blue]SQL [yellow]REPL")
    color = "blue"
    if not os.getenv("VIRTUAL_ENV"):
        color = "yellow"
        # c.print(f"[{color}]NOT IN VIRTUAL ENV")
    c.print(f"[dim]cwd:  [{color}]{disp_cwd} {warns}")
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    c.print(f"[dim]Py:   [{color}]{sys_executable} ({py_version})")
    c.print(f"[dim]repl: [{color}]{repl_executable} ")
    c.print(f"[dim]site: [{color}]{sitepackages}[/] ({has_customize}customized[/])")
    if loaded_sc := sys.modules.get("sitecustomize"):
        c.print(f"[dim]sitecustomize: [green]{loaded_sc.__file__.replace(homedir,'~')}[/]")  # type: ignore
    if loaded_uc := sys.modules.get("usercustomize"):
        c.print(f"[dim]usercustomize loaded from: [green]{loaded_uc.__file__.replace(homedir,'~')}[/]")  # type: ignore
    if ENABLE_USER_SITE:
        usersite = site.getusersitepackages().replace(homedir, "~")
        c.print(f"[dim][yellow]usersite: [green]{usersite}[/]")
    if not os.getenv("SQLREPL_NO_HELP"):
        c.print("\nType [red]help[/] for instructions and available commands")
    c.rule()

    log = logging.getLogger()
    if verbose:
        log.setLevel(logging.INFO)
    coroutine = embed(
        parent_globals=top_globals,
        parent_locals=top_locals,
        return_asyncio_coroutine=run_async,
        pipe_logs=pipe_logs,
    )
    if coroutine:
        asyncio.run(run_asyncio_coroutine(coroutine))


# if __name__ == "__main__":
# cli()
