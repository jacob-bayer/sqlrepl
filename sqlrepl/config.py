import os
import sys
from functools import lru_cache

ENV_REQUIRED_KEYS = ("PROJECT_ID", "DATASET_ID", "DEC_DATASET_ID", "VOLTAGE_DATASET")


@lru_cache
def get_jinja_params() -> dict[str, str]:
    try:
        project_id = os.environ["PROJECT_ID"]
        dataset_id = os.environ["DATASET_ID"]
        dec_dataset_id = os.environ["DEC_DATASET_ID"]
        voltage_dataset = os.environ["VOLTAGE_DATASET"]
    except KeyError:
        print(
            "Must set PROJECT_ID, DATASET_ID, DEC_DATASET_ID, & VOLTAGE_DATASET environment variables to start SQLREPL"
        )
        sys.exit(1)

    return dict(
        ENV="dev",
        PROJECT_ID=project_id,
        DATASET_ID=f"{project_id}.{dataset_id}",
        DEC_DATASET_ID=f"{project_id}.{dec_dataset_id}",
        VOLTAGE_DATASET=f"{project_id}.{voltage_dataset}",
    )
