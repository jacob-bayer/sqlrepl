import os
import site
from site import ENABLE_USER_SITE
from pathlib import Path
import sys
import asyncio
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
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.formatted_text import HTML, AnyFormattedText

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

pandas_bq_types = {k: pd.ArrowDtype(v) for k,v in pandas_bq_types.items()}


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
        DATASET_ID=os.environ["DATASET_ID"],
        DEC_DATASET_ID=os.environ["DEC_DATASET_ID"],
        VOLTAGE_DATASET=os.environ["VOLTAGE_DATASET"],
    )
except KeyError as e:
    print(
        "Must set PROJECT_ID, DATASET_ID, DEC_DATASET_ID, & VOLTAGE_DATASET environment variables to start SQLREPL"
    )
    sys.exit(1)


sqlkeywords = [
    "SELECT",
    "EXPORT",
    "DECLARE",
    "SET",
    "MERGE",
    "CALL",
    "WITH",
    "CREATE",
    "DROP",
    "INSERT",
    "UPDATE",
    "DELETE",
    "ASSERT",
    "ALTER",
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
            "SQL": "ansiblue",
            "Py": "ansigreen",
            "Debug": "ansired",
            "Async": "ansired",
            "DebugAsync": "ansired",
        }
        self.color = colors.get(prompt_title, "ansiblue")

    def in_prompt(self) -> AnyFormattedText:
        idx = self.python_input.current_statement_index
        title = self.prompt_title
        return HTML(f"<{self.color}>{title} [{idx}]</{self.color}>: ")

    def in2_prompt(self, width: int) -> AnyFormattedText:
        return "...: ".rjust(width)

    def out_prompt(self) -> AnyFormattedText:
        idx = self.python_input.current_statement_index
        color = "ansired"
        return HTML(f"<{color}>Result [{idx}]</{color}>: ")


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
        self.PROJECT_ID = jinja_params["PROJECT_ID"]
        self.DATASET_ID = jinja_params["DATASET_ID"]
        # self.NVIM_LISTEN_ADDRESS = os.environ["NVIM_SOCK"]
        # self._ensure_nvim()  # Initialize nvim context
        # self.c = Console()
        # self.get_globals = self.get_globals
        # self.get_locals = self.get_locals

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
                df = res.to_dataframe(bqstorage_client=self.storage_client, **pandas_bq_types) # type: ignore
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
        else:
            return "[bold][green]Query executed successfully"

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

        if filetype not in ["sql", "python"]:
            return

        if filetype == self.prompt_style:
            return

        if filetype == "sql":
            self.prompt_style = "sql"
            self._lexer = PygmentsLexer(GoogleSqlLexer)

        if filetype == "python":
            self.prompt_style = "python"
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
        if t.view_query:
            viewquery = format_fix(t.view_query) if not "insight" in t.dataset_id else t.view_query
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

        dofunc = {"sql": self.do_sql, "python": self.do_python}
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
        c.print(f"[dim]sitecustomize: [green]{loaded_sc.__file__.replace(homedir,'~')}[/]") # type: ignore
    if loaded_uc := sys.modules.get("usercustomize"):
        c.print(f"[dim]usercustomize loaded from: [green]{loaded_uc.__file__.replace(homedir,'~')}[/]") # type: ignore
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
