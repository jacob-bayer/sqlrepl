import os
import site
from jupyter_client.blocking.client import BlockingKernelClient
from site import ENABLE_USER_SITE
from pathlib import Path
import sys
import asyncio
from datetime import datetime, date
import logging
from tqdm import tqdm
import db_dtypes
from pygments.lexers.sql import GoogleSqlLexer
from pygments.lexers.python import PythonLexer
import pyarrow as pa
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit import print_formatted_text

# from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard
import pyperclip as pc
from sqlrepl.osc_clipboard import OSCClipboard
import ptpython
from ptpython.repl import PythonRepl

# from ptpython.ipython import InteractiveShellEmbed
from sqlrepl.bigquery import BigQuerySession
from sqlrepl.constants import SQL_KEYWORD_SET
from sqlrepl.formatting import format_fix
from sqlrepl.keybindings import register_keybindings
from sqlrepl.status_bar import status_bar
from sqlrepl.style import mystyle
from sqlrepl.ui import MyPrompt, printdf, showhelp
import click
from rich import print, get_console, inspect as ins

insm = lambda x: ins(x, methods=True)
insa = lambda x: ins(x, all=True)
from rich.console import Console
from rich.pretty import pprint
from rich.logging import RichHandler
from rich.progress import track

# from rich.markdown import Markdown
import google.cloud.bigquery as bq
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


# pretty.install()

os.environ["MANPAGER"] = "bat --language=py -p"
os.environ["PAGER"] = "bat --language=py -p"


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
        self.bq = BigQuerySession(self)
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

    def _render_dataframe_preview(self, df: pd.DataFrame) -> None:
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

    def _store_dataframe(self, df: pd.DataFrame) -> None:
        globals = self.get_globals()
        df.attrs["meta"] = {k: str(v) for k, v in df.dtypes.items()}
        globals["df"] = df
        self.dfs.append(df)
        globals["dfs"] = self.dfs

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
        self.bq.ensure_clients()
        estmsg = self.bq.print_query_estimate(query)
        if not estmsg:
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
        total_seconds = int((query_job.ended - query_job.started).total_seconds())
        timestring = f"[dim]{total_seconds} secs"
        mins, secs = divmod(total_seconds, 60)
        if mins >= 1:
            timecolor = "red" if mins >= 5 else "yellow" if mins >= 3 else "dim"
            timestring = f"[{timecolor}]{mins} mins {secs} secs[/]"
        self.c.print(f"[dim]{estmsg} --> {timestring} --> Done ([{color}]{bytes_billed}[/] GB)")
        # client.query_and_wait is also an option
        if is_select and res.total_rows:
            large_result = res.total_rows > 40000
            if self.bq_storage_mode or (is_linux and large_result):
                df = res.to_dataframe(bqstorage_client=self.storage_client, **pandas_bq_types)  # type: ignore
            else:
                globals = self.get_globals()
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
            rows, cols = df.shape
            self._render_dataframe_preview(df)
            self._store_dataframe(df)
            return f"Returned {rows} rows, {cols} cols"
        elif is_select:
            return "[bold][green]Query returned zero rows[/]"
        else:
            return f"[bold][green]Finished[/]"

    def _checkrunning(self):
        return self.bq.check_running()

    def handle_choice(self, filetype):

        if filetype not in ["sql", "python", "ipython"]:
            return

        if filetype == self.prompt_style:
            return

        if filetype == "sql":
            self.prompt_style = "sql"
            self._lexer = PygmentsLexer(GoogleSqlLexer)

        if filetype in ["python", "ipython"]:
            self.prompt_style = filetype
            self._lexer = PygmentsLexer(PythonLexer)

        return f"Mode: {self.prompt_style.upper()}"

    def lookup(self, tablename):
        self.bq.lookup_table(tablename)

    def _accept_handler(self, buff):
        words = buff.text.strip().split()
        if words and words[0] in SQL_KEYWORD_SET:
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
            self.bq.ensure_clients()
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
            elif isinstance(output, (list, dict, set)) and len(output) > 100:
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

    register_keybindings(repl)

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
