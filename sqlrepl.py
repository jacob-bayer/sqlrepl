import os
import sys
import json
from datetime import datetime, date
import contextlib
import logging
from tqdm import tqdm
from ollama import chat
from prompt_toolkit.filters import ViNavigationMode
from pygments.lexers.sql import GoogleSqlLexer
from pygments.lexers.python import PythonLexer
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.formatted_text import HTML, AnyFormattedText
import ptpython
from ptpython.prompt_style import PromptStyle
from ptpython.repl import PythonRepl
from ptpython.python_input import PythonInput
# from prompt_toolkit.patch_stdout import patch_stdout as patch_stdout_context

log = logging.getLogger("sqlrepl")

from pynvim import attach

try:
    import pyperclip as pc  # pyright: ignore

    pc.copy("")
except:

    class pc:
        @classmethod
        def copy(self, something):
            log.warning("pyperclip failed to import. Nothing copied.")
            return None


import pandas as pd

pd.options.display.max_rows = 4000
from decimal import Decimal
import sqlfluff
from zoneinfo import ZoneInfo
from rich import print, get_console, inspect as ins
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.errors import NotRenderableError
import google.cloud.bigquery as bq
from google.api_core.exceptions import GoogleAPICallError


def enable_logging():
    logging.disable(logging.NOTSET)


# pretty.install()
eastern = ZoneInfo("US/Eastern")


# import sqlite3

# database = 'jacobdb.db'

# os.environ['MANPAGER'] ="bat --language=python --theme=Dracula"
if "MANPAGER" in os.environ:
    del os.environ["MANPAGER"]
os.environ["PAGER"] = "less"


def help(someobj):
    print(someobj.__doc__)


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

    c = get_console()
    c.print(table)


class MyPrompt(PromptStyle):

    def __init__(self, python_input: PythonInput, prompt_title: str) -> None:
        self.python_input = python_input
        self.prompt_title = prompt_title
        colors = {
            "SQL   ": "ansiblue",
            "Python": "ansigreen",
            "LLM": "ansiyellow",
            "Debug": "ansired",
        }
        self.color = colors.get(prompt_title, "ansiblue")

    def in_prompt(self) -> AnyFormattedText:
        idx = self.python_input.current_statement_index
        title = self.prompt_title
        return HTML(f"<{self.color}>{title}[{idx}]</{self.color}>: ")

    def in2_prompt(self, width: int) -> AnyFormattedText:
        return "...: ".rjust(width)

    def out_prompt(self) -> AnyFormattedText:
        idx = self.python_input.current_statement_index
        color = "red"
        return HTML(f"<{color}>Result[{idx}]</{color}>: ")


class MyRpl(PythonRepl):

    def _ensure_nvim(self) -> bool:
        print = self.c.print
        if not os.path.exists(self.NVIM_LISTEN_ADDRESS):
            return False

        new = ""
        if self.nvim:
            try:
                self.nvim.current.buffer
                return True
            except:
                self.nvim = None
                new = "new"

        if not self.nvim:
            try:
                self.nvim = attach("socket", path=self.NVIM_LISTEN_ADDRESS)
                tmux_pane = os.getenv("TMUX_PANE")
                print(f"Attached to {new} NVIM")
                if tmux_pane:
                    os.system(f"tmux setenv -g TMUX_TARGET_ID ''{tmux_pane}''")
                    self.nvim.api.set_var("tmux_target_id", tmux_pane)
                return True
            except Exception as e:
                print("Error attaching to nvim:", e)

        return False

    def __init__(self, debug_mode=False, *args, **kwargs) -> None:
        kwargs["vi_mode"] = True
        kwargs["history_filename"] = os.environ["HOME"] + "/ptpython_history_sql"
        super().__init__(*args, **kwargs)
        self.style = "dracula"
        self.debug_mode = debug_mode
        if os.environ.get("OS_THEME") == "light":
            self.style = "default"
        self.use_code_colorscheme(self.style)
        self._lexer = PygmentsLexer(PythonLexer)
        self.all_prompt_styles["sql"] = MyPrompt(self, "SQL")
        self.all_prompt_styles["python"] = MyPrompt(self, "Python")
        self.all_prompt_styles["llm"] = MyPrompt(self, "LLM")
        if self.debug_mode:
            self.all_prompt_styles["python"] = MyPrompt(self, "Debug")
        self.prompt_style = "python"
        self.confirm_exit = True
        self.enable_input_validation = False
        self.show_docstring = True
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
        self.messages = [
            {
                "role": "system",
                "content": "You are a code expert. If you respond to the user with a code snippet, denote this using three backticks followed by the language (ex: ```python\n\nimport os\n```",
            }
        ]
        self.client = None
        self.PROJECT_ID = ""
        self.DATASET_ID = ""
        self.NVIM_LISTEN_ADDRESS = os.environ["NVIM_SOCK"]
        self._ensure_nvim()  # Initialize nvim context
        # self.c = Console()
        # self.get_globals = self.get_globals
        # self.get_locals = self.get_locals

    def _get_bq_client(self) -> tuple[bq.Client, bq.Client]:

        print("[blue]Establishing connection to BigQuery")

        # Instantiating client does this step anyway so it doesnt add any extra time
        from google.auth import default

        credentials, project = default()

        # Use PROJECT_ID env var if set, otherwise project from credentials
        PROJECT_ID = os.getenv("PROJECT_ID", project)
        if not PROJECT_ID:
            raise Exception("Please set DATASET_ID, DEC_DATASET_ID in the environment")
        if isinstance(PROJECT_ID, str):
            self.PROJECT_ID = PROJECT_ID
            print("[blue]Project ID:", self.PROJECT_ID)

        try:
            self.DATASET_ID = os.environ["DATASET_ID"]
            self.DEC_DATASET_ID = os.environ["DEC_DATASET_ID"]
        except KeyError:
            raise Exception("Please set DATASET_ID, DEC_DATASET_ID in the environment")

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

    def llm(self, line: str) -> object:
        print = self.c.print
        self.messages.append({"role": "user", "content": line})
        full_response = ""
        # think = False
        # counter = 0
        for part in chat("deepseek-r1:7b", stream=True, messages=self.messages):
            response = part["message"]["content"]
            # ishtml = response.startswith("<") and response.endswith(">")
            # if ishtml:
            #     think = response == '<think>'
            #     response = ""
            #     if think:
            #         self.c.rule("[bold red]Thinking...")
            #     else:
            #         self.c.rule("[bold red]")
            print(response, end="")
            full_response += response

        new_full_response = []
        for line in full_response.split("\n"):
            if "```" in line:
                line = line.replace(" ", "")
            new_full_response.append(line)
        full_response = "\n".join(new_full_response)

        # if full_response:
        print("\n\n")
        self.messages += [{"role": "assistant", "content": full_response}]

        full_response = Markdown(full_response)

        for token in full_response.parsed:
            if token.tag == "code":
                pc.copy(token.content)

        self.c.rule("[bold red]Assistant")
        return full_response

    def do_python(self, line: str):

        globals = self.get_globals()
        try:
            # if self._ensure_nvim():
                # This causes an issue with dask distributed unless it's protected by __name__==__main__
                # globals["__file__"] = self.nvim.current.buffer.name
            output = super().eval(line)
            # if output is not None:
            # self.c.print(output)
            return output
        except Exception as e:
            globals["last_exception"] = e
            globals["last_frame"] = sys
            return e
            # get_console().print_exception(
            # show_locals=False, suppress=[ptpython], max_frames=10
            # )

    def do_sql(self, line: str) -> object:
        if not self.client:
            self.client, self.dry_client = self._get_bq_client()

        print = self.c.print

        globals = self.get_globals()

        query = line
        query = query.replace("{{ENV}}", "dev")
        query = query.replace("{{PROJECT_ID}}", self.PROJECT_ID)
        query = query.replace("{{DATASET_ID}}", self.DATASET_ID)
        query = query.replace("{{DEC_DATASET_ID}}", self.DEC_DATASET_ID)
        query = query.replace("{{VOLTAGE_DATASET}}", "voltage_anbc_hcb_dev")
        query = sqlfluff.fix(query, config_path=os.environ["HOME"] + "/.sqlfluff")
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
        globals["running_jobs"] = rj = list(
            self.client.list_jobs(state_filter="running")
        )
        if rj:
            return "There are running jobs. Check `running_jobs`"

    def handle_choice(self, filetype):

        if filetype not in ["sql", "python", "llm"]:
            return

        if filetype == self.prompt_style:
            return

        if filetype == "sql":
            self.prompt_style = "sql"
            self._lexer = PygmentsLexer(GoogleSqlLexer)

        if filetype == "llm":
            self.prompt_style = "llm"
            self._lexer = None

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

        tablename = tablename.replace("{{ENV}}", "dev")
        tablename = tablename.replace("{{PROJECT_ID}}", self.PROJECT_ID)
        tablename = tablename.replace("{{DATASET_ID}}", self.DATASET_ID)
        tablename = tablename.replace("{{DEC_DATASET_ID}}", self.DEC_DATASET_ID)
        tablename = tablename.replace("{{VOLTAGE_DATASET}}", "voltage_anbc_hcb_dev")

        t = self.client.get_table(tablename)
        print(f"\n{t.reference}\n")
        print(f"Type: {t.table_type}")
        if t.view_query:
            viewquery = sqlfluff.fix(
                t.view_query, config_path="/Users/n856925/.sqlfluff"
            )
            highlighted = Syntax(
                viewquery, "googlesql", theme=self.style, line_numbers=True
            )
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
            modified = t.modified.astimezone(eastern).strftime(
                "%Y-%m-%d %I:%M:%S %p ET"
            )
            print(f"Modified at: {modified}")
        if t.created:
            created = t.created.astimezone(eastern).strftime("%Y-%m-%d %I:%M:%S %p ET")
            print(f"Created at: {created}")

        globals = self.get_globals()
        globals["t"] = t
        print("\nTable object is globally assigned to `t` for exploration\n")

    def eval(self, line: str) -> object:

        choice = self.handle_choice(line)
        if choice:
            return choice

        if line == "help":
            allcommands = ["checkrunning", "sql", "python"]
            self.c.print("Commands:\n", "\n".join(allcommands))

        if line in ["checkrunning", "getjobs"]:
            return self._checkrunning()

        if line == "llm history":
            for message in self.messages:
                self.c.print("\n")
                self.c.rule(f"[bold red]{message['role']}")
                self.c.print(Markdown(message["content"]), highlight=True)
                self.c.print("\n")
            return

        if line == "reset_nvim_tries":
            self.nvim = None
            self._ensure_nvim()
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
        elif not self.prompt_style == "llm":
            self.handle_choice("python")

        dofunc = {"sql": self.do_sql, "llm": self.llm, "python": self.do_python}
        output = dofunc[self.prompt_style](line)
        has_output = output is not None
        # The best strategy would be to learn how self.style_transformations works in conjunction
        # with the parent class output printer.
        if has_output:
            if isinstance(output, Exception):
                raise output
                # try:
                    # raise output
                # except Exception as e:
                    # get_console().print_exception(
                        # show_locals=False, suppress=[ptpython], max_frames=10
                    # )
                return
            print(output)

        # if isinstance(output, pd.DataFrame):
        # print(output)
        # return
        # else:
        # return output


# The globals and locals have to be set up exactly this way. Exactly nested.
# Otherwise it doesn't work.


def embed(
    debug_mode=True,
    parent_globals=None,
    parent_locals=None,
    return_asyncio_coroutine=False,
):

    def get_globals():
        if parent_globals is None:
            return {}
        return parent_globals

    def get_locals():
        if parent_locals is None:
            return {}
        return parent_locals

    # if not debug_mode:
    # logging.disable()

    # Create REPL.
    repl = MyRpl(debug_mode=debug_mode, get_globals=get_globals, get_locals=get_locals)

    @repl.add_key_binding("E", filter=ViNavigationMode())
    def _(event) -> None:
        event.current_buffer.cursor_position += (
            event.current_buffer.document.get_end_of_line_position()
        )

    # @repl.add_key_binding("f8")
    # def _(event) -> None:
    #     if event.app.current_buffer.text:
    #         breakpoint()
    #         event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

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
        b.cursor_position += b.document.get_start_of_line_position(
            after_whitespace=True
        )

    @repl.add_key_binding("c-space")
    def _(event) -> None:
        """
        Accept suggestion.
        """
        b = event.current_buffer
        suggestion = b.suggestion

        if suggestion:
            b.insert_text(suggestion.text)

    # Start repl.
    # patch_context = patch_stdout_context()
    #
    # if return_asyncio_coroutine:
    #
    #     async def coroutine() -> None:
    #         with patch_context:
    #             await repl.run_async()
    #
    #     return coroutine()  # type: ignore
    # else:
    repl.run()
        # return None


if __name__ == "__main__":
    embed(
        debug_mode=False,
        parent_globals=globals(),
        parent_locals=locals(),
        return_asyncio_coroutine=False,
    )
    # if x:
        # asyncio.run(x)  # type: ignore
