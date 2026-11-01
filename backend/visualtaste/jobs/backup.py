"""Back up, list, verify and restore the database.

    python -m visualtaste.jobs.backup                    # snapshot + JSONL export now
    python -m visualtaste.jobs.backup --list             # snapshots, newest first, with row counts
    python -m visualtaste.jobs.backup --restore latest   # or a snapshot's file name

Stop the API before restoring: it holds the database open. A restore first
snapshots the current database, so it can be undone with another restore,
and it prints the restored row counts beside the snapshot's for comparison.
The API also snapshots on shutdown and once a day while running.
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--restore", metavar="SNAPSHOT", default=None)
    args = parser.parse_args(argv)

    from visualtaste.config import get_settings
    from visualtaste.db.backup import (
        list_snapshots,
        restore_database,
        run_backup,
        table_counts,
    )

    settings = get_settings()
    snapshots = list_snapshots(settings)

    if args.list:
        for path in snapshots:
            print(path.name, table_counts(path))
        return 0

    if args.restore:
        chosen = snapshots[0] if args.restore == "latest" and snapshots else None
        chosen = chosen or next((p for p in snapshots if p.name == args.restore), None)
        if chosen is None:
            print(f"no snapshot {args.restore!r}; see --list", file=sys.stderr)
            return 2
        expected = table_counts(chosen)
        restored = restore_database(settings, chosen)
        print("snapshot:", expected)
        print("restored:", restored)
        if expected != restored:
            print("ROW COUNTS DIFFER", file=sys.stderr)
            return 1
        print("row counts match")
        return 0

    snapshot, jsonl = run_backup(settings)
    print(f"snapshot: {snapshot}\nevents JSONL: {jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
