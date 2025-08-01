from prompt_toolkit.styles import Style

mystyle = Style.from_dict({
    "control-character": "ansiblue",
    # Classic prompt.
    "prompt": "bold",
    "prompt.dots": "noinherit",
    # (IPython <5.0) Prompt: "In [1]:"
    "in": "bold #008800",
    "in.number": "",
    # Return value.
    "out": "#ff0000",
    "out.number": "#ff0000",
    # Completions.
    "completion.builtin": "",
    "completion.param": "#006666 italic",
    "completion.keyword": "fg:#008800",
    "completion.keyword fuzzymatch.inside": "fg:#008800",
    "completion.keyword fuzzymatch.outside": "fg:#44aa44",
    # Separator between windows. (Used above docstring.)
    "separator": "#bbbbbb",
    # System toolbar
    "system-toolbar": "#22aaaa noinherit",
    # "arg" toolbar.
    "arg-toolbar": "#22aaaa noinherit",
    "arg-toolbar.text": "noinherit",
    # Signature toolbar.
    "signature-toolbar": "bg:#44bbbb #000000",
    "signature-toolbar current-name": "bg:#008888 #ffffff bold",
    "signature-toolbar operator": "#000000 bold",
    "docstring": "#888888",
    # Validation toolbar.
    "validation-toolbar": "bg:#440000 #aaaaaa",
    # Status toolbar.
    # "status-toolbar": "bg:#0000B7 #aaaaaa",
    "status-toolbar": "bg:#222437 #808BBC",
    "status-toolbar.title": "underline",
    "status-toolbar.inputmode": "bg:#222222 #ffffaa",
    "status-toolbar.key": "bg:#000000 #888888",
    "status-toolbar key": "bg:#000000 #888888",
    "status-toolbar.pastemodeon": "bg:#aa4444 #ffffff",
    "status-toolbar.pythonversion": "bg:#222222 #ffffff bold",
    "status-toolbar paste-mode-on": "bg:#aa4444 #ffffff",
    "record": "bg:#884444 white",
    "status-toolbar more": "#ffff44",
    "status-toolbar.input-mode": "#ffff44",
    # The options sidebar.
    "sidebar": "bg:#bbbbbb #000000",
    "sidebar.title": "bg:#668866 #ffffff",
    "sidebar.label": "bg:#bbbbbb #222222",
    "sidebar.status": "bg:#dddddd #000011",
    "sidebar.label selected": "bg:#222222 #eeeeee",
    "sidebar.status selected": "bg:#444444 #ffffff bold",
    "sidebar.separator": "underline",
    "sidebar.key": "bg:#bbddbb #000000 bold",
    "sidebar.key.description": "bg:#bbbbbb #000000",
    "sidebar.helptext": "bg:#fdf6e3 #000011",
    #        # Styling for the history layout.
    #        history.line:                          '',
    #        history.line.selected:                 'bg:#008800  #000000',
    #        history.line.current:                  'bg:#ffffff #000000',
    #        history.line.selected.current:         'bg:#88ff88 #000000',
    #        history.existinginput:                  '#888888',
    # Help Window.
    "window-border": "#aaaaaa",
    "window-title": "bg:#bbbbbb #000000",
    # Meta-enter message.
    "accept-message": "bg:#ffff88 #444444",
    # Exit confirmation.
    "exit-confirmation": "bg:#884444 #ffffff",
})

