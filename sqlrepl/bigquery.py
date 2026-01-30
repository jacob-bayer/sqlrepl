from __future__ import annotations

import logging
from zoneinfo import ZoneInfo

import google.cloud.bigquery as bq
import google.cloud.bigquery_storage as bqs
from rich.syntax import Syntax

from sqlrepl.formatting import apply_jinja_params, format_fix

log = logging.getLogger(__name__)
eastern = ZoneInfo("US/Eastern")


class BigQuerySession:
    def __init__(self, repl) -> None:
        self.repl = repl

    def get_clients(self) -> tuple[bq.Client, bq.Client]:
        self.repl.c.print("[blue]Connecting...", end=" ")

        # Instantiating client does this step anyway so it doesnt add any extra time
        from google.auth import default

        credentials, _ = default()

        default_dataset = bq.DatasetReference(self.repl.PROJECT_ID, self.repl.DATASET_ID)
        default_config = bq.QueryJobConfig(
            default_dataset=default_dataset, job_timeout_ms=600000  # 10 min timeout
        )
        dry_run_config = bq.QueryJobConfig(
            default_dataset=default_dataset, dry_run=True, use_query_cache=False
        )
        client = bq.Client(
            credentials=credentials,
            default_query_job_config=default_config,
        )
        dry_client = bq.Client(
            project=self.repl.PROJECT_ID,
            credentials=credentials,
            default_query_job_config=dry_run_config,
        )
        self.repl.c.print("[green]Done", end="\r")

        if self.repl.bq_storage_mode:
            self.repl.storage_client = bqs.BigQueryReadClient(credentials=credentials)

        return client, dry_client

    def ensure_clients(self) -> None:
        if not self.repl.client:
            self.repl.client, self.repl.dry_client = self.get_clients()

    def normalize_table_name(self, tablename: str) -> str:
        splittable = tablename.split(".")
        if len(splittable) == 1:
            tablename = self.repl.PROJECT_ID + "." + self.repl.DATASET_ID + "." + tablename
        if len(splittable) == 2:
            tablename = self.repl.PROJECT_ID + "." + tablename
        return apply_jinja_params(tablename)

    def print_query_estimate(self, query: str) -> str | None:
        try:
            query_job = self.repl.dry_client.query(query)
            bytes_billed = round((query_job.total_bytes_processed or 0) / 1e9, 3)
            estmsg = f"Est {bytes_billed} GB"
            self.repl.c.print(f"[dim]{estmsg}", end="... ")
            return estmsg
        except Exception as e:
            self.repl.c.print(f"\n{e}")
            return None

    def check_running(self) -> str | None:
        globals = self.repl.get_globals()
        self.ensure_clients()
        globals["bqjobs"] = self.repl.bqjobs = list(self.repl.client.list_jobs(max_results=20))
        globals["running_jobs"] = self.repl.running_jobs = list(
            self.repl.client.list_jobs(state_filter="running")
        )
        if self.repl.running_jobs:
            return "There are running jobs. Check `running_jobs`"
        return None

    def lookup_table(self, tablename: str) -> None:
        self.ensure_clients()

        print = self.repl.c.print
        t = self.repl.client.get_table(self.normalize_table_name(tablename))
        print(f"\n{t.reference}\n")
        print(f"Type: {t.table_type}")
        if viewquery := t.view_query:
            try:
                viewquery = (
                    format_fix(t.view_query) if not "insight" in t.dataset_id else t.view_query
                )
            except KeyboardInterrupt:
                print("\n[red]View query display cancelled\n")
            highlighted = Syntax(viewquery, "googlesql", theme=self.repl.style, line_numbers=True)
            print("\n", highlighted, "\n")
        else:
            print(f"{t.num_rows} rows")
        print("Columns")
        colors = {
            "NUMERIC": "yellow",
            "FLOAT": "yellow",
            "INTEGER": "yellow",
            "STRING": "green",
            # "BOOLEAN": "magenta",
            "TIMESTAMP": "cyan",
            "DATE": "cyan",
        }
        for col in t.schema:
            color = colors.get(col.field_type, "reset")
            print(f"[{color}]{col.name} {col.field_type}")

        print("\n")

        if t.modified:
            modified = t.modified.astimezone(eastern).strftime("%Y-%m-%d %I:%M:%S %p ET")
            print(f"Modified at: {modified}")
        if t.created:
            created = t.created.astimezone(eastern).strftime("%Y-%m-%d %I:%M:%S %p ET")
            print(f"Created at: {created}")

        globals = self.repl.get_globals()
        globals["t"] = t
        print("\nTable object is globally assigned to `t` for exploration\n")
