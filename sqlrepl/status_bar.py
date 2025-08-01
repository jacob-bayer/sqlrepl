import os
import sys
from prompt_toolkit.filters import Condition, is_done
from prompt_toolkit.layout import ConditionalContainer, Window, AnyContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.formatted_text.base import StyleAndTextTuples
from prompt_toolkit.filters import renderer_height_is_known


def status_bar(python_input) -> AnyContainer:
    """
    Create the `Layout` for the status bar.
    """
    venv = os.getenv("VIRTUAL_ENV", "")
    venv = os.path.dirname(venv)
    venv = os.path.basename(venv)
    py = sys.version_info

    TB = "class:status-toolbar"

    def get_text_fragments() -> StyleAndTextTuples:
        python_buffer = python_input.default_buffer

        result: StyleAndTextTuples = []
        append = result.append

        if python_input.title:
            result.extend(to_formatted_text(python_input.title))

        # Position in history.
        # append(
        # (
        # TB,
        # "%i/%i "
        # % (python_buffer.working_index + 1, len(python_buffer._working_lines)),
        # )
        # )

        if venv:
            result.append((TB, f" ({venv}) "))
        result.append(
            (
                TB,
                f"{py.major}.{py.minor}.{py.micro}",
            )
        )
        return result

    return ConditionalContainer(
        content=Window(content=FormattedTextControl(get_text_fragments), style=TB),
        filter=~is_done
        & renderer_height_is_known
        & Condition(
            lambda: python_input.show_custom_status_bar
            and not python_input.show_exit_confirmation
        ),
    )
