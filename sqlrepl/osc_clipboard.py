import sys
import base64
from prompt_toolkit.clipboard import Clipboard, ClipboardData

class OSCClipboard(Clipboard):
    def __init__(self) -> None:
        self._data: ClipboardData | None = None

    def set_data(self, data: ClipboardData) -> None:
        self._data = data
        encoded_text = base64.b64encode(data.text.encode()).decode()
        osc_52 = f"\033]52;c;{encoded_text}\a"
        # Write it to the terminal
        sys.stdout.write(osc_52)
        sys.stdout.flush()


    def get_data(self) -> ClipboardData:
        if self._data:
            return self._data
        return ClipboardData()

