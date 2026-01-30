import logging
import os

from prompt_toolkit.filters import Condition, vi_navigation_mode
from prompt_toolkit.key_binding.vi_state import InputMode

log = logging.getLogger(__name__)


def register_keybindings(repl) -> None:
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
