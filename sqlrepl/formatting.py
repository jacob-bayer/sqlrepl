import os

import sqlfluff

from sqlrepl.config import get_jinja_params


def apply_jinja_params(text: str) -> str:
    for key, value in get_jinja_params().items():
        text = text.replace("{{" + key + "}}", value)
    return text


def format_fix(query: str) -> str:
    return sqlfluff.fix(apply_jinja_params(query), config_path=os.environ["HOME"] + "/.sqlfluff")
