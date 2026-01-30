import contextlib
from datetime import date, datetime
from decimal import Decimal

import pandas as pd
from prompt_toolkit.formatted_text import HTML, AnyFormattedText
from ptpython.prompt_style import PromptStyle
from ptpython.python_input import PythonInput
from rich import get_console
from rich.errors import NotRenderableError
from rich.table import Table

DF_DTYPE_COLORS = {
    "object": "blue",
    "float64": "green",
    "int64": "magenta",
    "bool": "cyan",
    "datetime64[ns]": "yellow",
}

DF_VALUE_COLORS = {
    str: "blue",
    float: "green",
    int: "magenta",
    bool: "cyan",
    date: "yellow",
    datetime: "yellow",
    pd.Timestamp: "yellow",
    Decimal: "green",
}


def showhelp() -> None:
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

    colors = DF_DTYPE_COLORS

    rowcolor = False
    if color_by != "dtype":
        colors = DF_VALUE_COLORS
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
