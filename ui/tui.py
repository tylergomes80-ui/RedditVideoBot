from rich.live import Live
from rich.table import Table
from time import sleep
from pathlib import Path

def run_tui():
    table = Table(title="HorrorBot 2.0 Dashboard")
    table.add_column("Metric"); table.add_column("Value")
    with Live(table, refresh_per_second=4) as live:
        for i in range(20):
            table.rows = []
            table.add_row("Jobs queued", str(max(0,10-i)))
            table.add_row("Jobs done", str(i))
            table.add_row("CPU", f"{30+i}%")
            live.update(table)
            sleep(0.2)
