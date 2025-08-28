import os
from pathlib import Path
import sys
import asyncio
from datetime import datetime, date
import contextlib
import logging
from tqdm import tqdm
from prompt_toolkit.filters import ViNavigationMode
from pygments.lexers.sql import GoogleSqlLexer
from pygments.lexers.python import PythonLexer
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.formatted_text import HTML, AnyFormattedText
import ptpython
from ptpython.prompt_style import PromptStyle
from ptpython.repl import PythonRepl
from ptpython.python_input import PythonInput
from sqlrepl.status_bar import status_bar
from sqlrepl.style import mystyle
import click


# venv = os.getenv("VIRTUAL_ENV")
# sys.path
# if venv:
# sys.path.insert(1, venv + "/lib/python3.11/site-packages")


log = logging.getLogger("sqlrepl")

# from pynvim import attach

try:
    import pyperclip as pc  # pyright: ignore

    pc.copy("")
except:

    class pc:
        @classmethod
        def copy(cls, something):
            log.warning("pyperclip failed to import. Nothing copied.")
            return None


import pandas as pd
import numpy as np

pd.options.display.max_rows = 4000
from decimal import Decimal
import sqlfluff

logging.getLogger("sqlfluff").setLevel(logging.WARNING)

jinja_params = dict(
    ENV="dev",
    PROJECT_ID=os.environ["PROJECT_ID"],
    DATASET_ID=os.environ["DATASET_ID"],
    DEC_DATASET_ID=os.environ["DEC_DATASET_ID"],
    VOLTAGE_DATASET="voltage_anbc_hcb_dev",
)


def format_fix(query):
    for key, value in jinja_params.items():
        query = query.replace("{{" + key + "}}", value)
    return sqlfluff.fix(query, config_path=os.environ["HOME"] + "/.sqlfluff")


from zoneinfo import ZoneInfo
from rich import print, get_console, inspect as ins
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.errors import NotRenderableError
import google.cloud.bigquery as bq
from google.api_core.exceptions import GoogleAPICallError


# pretty.install()
eastern = ZoneInfo("US/Eastern")

os.environ["MANPAGER"] = "bat --language=py -p"
# if "MANPAGER" in os.environ:
# del os.environ["MANPAGER"]
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
        "Int64": "magenta",
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
            rowtype = type(row[color_by])
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
        color = "red"
        return HTML(f"<{color}>Result [{idx}]</{color}>: ")


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

    def __init__(
        self,
        title: str | None = None,
        debug_mode: bool = False,
        is_async: bool = False,
        *args,
        **kwargs,
    ) -> None:
        kwargs["vi_mode"] = True
        kwargs["history_filename"] = os.environ["HOME"] + "/ptpython_history_sql"
        kwargs["_extra_toolbars"] = [status_bar(self)]
        super().__init__(*args, **kwargs)
        self.debug_mode = debug_mode
        self.async_loop = is_async
        self.async_debug = self.debug_mode and self.async_loop
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
        self.show_docstring = True
        self.show_status_bar = False
        self.ptpython_layout.status_bar = status_bar
        # self.clipboard = PyperclipClipboard()
        self.terminal_title = "OMREPL"
        self.enable_open_in_editor = False
        self.enable_auto_suggest = True
        self.enable_history_search = True
        self.enable_output_formatting = True
        self.enable_pager = True
        self.complete_while_typing = False
        self.wrap_lines = True
        self.show_line_numbers = True
        self.highlight_matching_parenthesis = True
        self.c = Console()
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

        print("[blue]Establishing connection to BigQuery")

        # Instantiating client does this step anyway so it doesnt add any extra time
        from google.auth import default

        credentials, _ = default()

        default_dataset = bq.DatasetReference(self.PROJECT_ID, self.DATASET_ID)
        default_config = bq.QueryJobConfig(default_dataset=default_dataset)
        dry_run_config = bq.QueryJobConfig(
            default_dataset=default_dataset, dry_run=True, use_query_cache=False
        )
        client = bq.Client(
            project=self.PROJECT_ID,
            credentials=credentials,
            default_query_job_config=default_config,
        )
        dry_client = bq.Client(
            project=self.PROJECT_ID,
            credentials=credentials,
            default_query_job_config=dry_run_config,
        )
        print("[green]Connected to BigQuery")

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
            globals["last_frame"] = sys
            get_console().print_exception(show_locals=False, suppress=[ptpython], max_frames=10)

    def do_sql(self, line: str) -> object:
        if not self.client:
            self.client, self.dry_client = self._get_bq_client()

        print = self.c.print

        globals = self.get_globals()

        query = format_fix(line)
        syntax = Syntax(query, "googlesql", theme=self.style, line_numbers=True)
        print(syntax)

        try:
            query_job = self.dry_client.query(query)
            bytes_billed = round((query_job.total_bytes_processed or 0) / 1e9, 3)
            print("Est: ", bytes_billed, "GB")
        except Exception as e:
            print(f"\n{e}")
            return
        print("[yellow]Working...")
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

        pc.copy(query)
        bytes_billed = round((query_job.total_bytes_billed or 0) / 1e9, 3)
        print("Actual: ", bytes_billed, "GB")
        # client.query_and_wait is also an option
        is_select = line.startswith(("SELECT", "WITH"))
        if is_select:
            # df = query_job.to_dataframe()
            globals["rows"] = []
            globals["dictrows"] = []
            for row in tqdm(res, total=res.total_rows):
                globals["rows"].append(row)
                globals["dictrows"].append(dict(row))
            df = pd.DataFrame(globals["dictrows"])

            rows, cols = df.shape
            if rows == 1:
                print(df.T)
            elif cols > 10:
                print(df.head(10))
            else:
                print("\n")
                printdf(df.head(20))
                # print(Markdown(df.to_markdown()))
                print("\n")
            # globals["df"] = self.d.data = df
            globals["df"] = df
            self.dfs.append(df)
            globals["dfs"] = self.dfs
            return f"Returned {rows} rows, {cols} cols"
        else:
            return "Query executed successfully"

    def _checkrunning(self):
        globals = self.get_globals()
        if not self.client:
            self.client, self.dry_client = self._get_bq_client()
        globals["bqjobs"] = list(self.client.list_jobs(max_results=20))
        globals["running_jobs"] = rj = list(self.client.list_jobs(state_filter="running"))
        if rj:
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
            viewquery = format_fix(t.view_query)
            highlighted = Syntax(viewquery, "googlesql", theme=self.style, line_numbers=True)
            print(f":\n", highlighted, "\n")
        else:
            print(f"{t.num_rows} rows")
        print("Columns")
        colors = {
            "NUMERIC": "blue",
            "INTEGER": "blue",
            "STRING": "green",
            "Int64": "magenta",
            "bool": "cyan",
            "TIMESTAMP": "yellow",
            "DATE": "yellow",
        }
        for col in t.schema:
            color = colors.get(col.field_type, "")
            print(f"[{color}]{col.field_type}: {col.name}")

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

    async def eval_async(self, line: str) -> object:
        try:
            x = await super().eval_async(line)
            sys.stdout.flush()
            if x is not None:
                print(x)
        except Exception as e:
            log.exception(e)

    def eval(self, line: str) -> object:

        choice = self.handle_choice(line)
        if choice:
            return choice

        if line == "help":
            allcommands = ["checkrunning", "sql", "python"]
            self.c.print("Commands:\n", "\n".join(allcommands))

        if line in ["checkrunning", "getjobs"]:
            return self._checkrunning()

        if line == "reset_nvim_tries":
            # self.nvim = None
            # self._ensure_nvim()

            tmux_pane = os.getenv("TMUX_PANE")
            if tmux_pane:
                os.system(f"tmux setenv -g TMUX_TARGET_ID ''{tmux_pane}''")

            return "Reset nvim tries"

        if line.startswith("lookup"):
            self.lookup(line.split()[-1].replace("`", ""))
            return

        issql = line.split()[0] in [
            "SELECT",
            "WITH",
            "CREATE",
            "DROP",
            "INSERT",
            "UPDATE",
            "DELETE",
            "ASSERT",
            "ALTER",
        ]

        if issql:
            self.handle_choice("sql")
        else:
            self.handle_choice("python")

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
                    get_console().print_exception(
                        show_locals=False,
                        suppress=[ptpython],
                        max_frames=10,
                        extra_lines=10,
                    )
                return
            print(output)

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
    )

    @repl.add_key_binding("E", filter=ViNavigationMode())
    def _(event) -> None:
        event.current_buffer.cursor_position += (
            event.current_buffer.document.get_end_of_line_position()
        )

    @repl.add_key_binding("f8")
    def _(event) -> None:
        if event.app.current_buffer.text:
            breakpoint()
            event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    # To debug events do this ( if in the repl, do get_globals = globals )
    # globals = get_globals()
    # globals['event'] = event
    # Then the event can be explored in the repl
    # Slight modification of the default ctrl-c so that if there's no text it doesn't do anything
    @repl.add_key_binding("c-c")
    def _(event) -> None:
        if event.app.current_buffer.text:
            event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    # can also use "is_multiline" condition to make ] and [ work when not in multiline
    # https://github.com/prompt-toolkit/ptpython/blob/5021832f76309755097b744f274c4e687a690b85/ptpython/key_bindings.py
    @repl.add_key_binding("c-y")
    def _(event) -> None:
        os.system("tmux copy-mode")

    @repl.add_key_binding("c-u")
    def _(event) -> None:
        os.system("tmux copy-mode")

    # press gf to go to the file under cursor
    # '/Users/n856925/Documents/github/ptpython/ptpython/key_bindings.py'
    @repl.add_key_binding("c-q", filter=ViNavigationMode())
    def _(event) -> None:
        if repl.confirm_exit:
            # Show exit confirmation and focus it (focusing is important for
            # making sure the default buffer key bindings are not active).
            repl.show_exit_confirmation = True
            repl.app.layout.focus(repl.ptpython_layout.exit_confirmation)
        else:
            event.app.exit(exception=EOFError)

    @repl.add_key_binding("B", filter=ViNavigationMode())
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
    paths = {
        "not a git repo": Path(".git"),
        "not a pyproject": Path("pyproject.toml")
    }
    warnlist = [message for message, path in paths.items() if not path.exists()]
    return f"[yellow]({', '.join(warnlist)})" if warnlist else ""

@click.command()
@click.option("--run_async", is_flag=True, help="Async")
@click.option("--verbose", is_flag=True, help="INFO output")
# @click.option("--info", is_flag=True, help="Info output")
def cli(run_async, verbose):
    """Command-line interface for the embed function."""

    # stdout bc otherwise there's softwrap
    getsitecmd = "python -c 'import site, sys; sys.stdout.write(site.getsitepackages()[0])'"
    sitepackages = os.popen(getsitecmd).read().strip()
    sys.path.insert(0, sitepackages)

    getreplcmd = "which sqlrepl"
    repl_executable = os.popen(getreplcmd).read().strip()

    cwd = disp_cwd = os.getcwd()
    homedir = os.path.expanduser("~")
    if os.path.exists(homedir):
        disp_cwd = cwd.replace(homedir, "~")
    rel_sitepackages = os.path.relpath(sitepackages, cwd)
    rel_repl = os.path.relpath(repl_executable, cwd)
    rel_executable = os.path.relpath(sys.executable, cwd)
    sitecustomize = os.path.join(rel_sitepackages, "sitecustomize.py")
    has_customize = "[red]not " if not os.path.isfile(sitecustomize) else "[green]"
    no_vcs = "not a git repo" if not os.path.exists("./.git") else ""
    no_proj = "not a pyproject" if not os.path.exists("./pyproject.toml") else ""
    warnlist = [no_vcs, no_proj]
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
    c.print(f"[dim]Py:   [{color}]{rel_executable} ({py_version})")
    c.print(f"[dim]repl: [{color}]{rel_repl}")
    c.print(f"[dim]site: [{color}]{rel_sitepackages}[/] ({has_customize}customized[/])")
    c.rule()

    log = logging.getLogger()
    if verbose:
        log.setLevel(logging.INFO)
    coroutine = embed(
        parent_globals=top_globals,
        parent_locals=top_locals,
        return_asyncio_coroutine=run_async,
    )
    if coroutine:
        asyncio.run(run_asyncio_coroutine(coroutine))


if __name__ == "__main__":
    cli()
