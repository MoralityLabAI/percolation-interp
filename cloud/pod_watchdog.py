"""Delete a Prime pod at a deadline, so a forgotten or stuck pod cannot drain the budget.

  python cloud/pod_watchdog.py POD_ID DEADLINE_FILE      (prime CLI venv's python; launch detached)

DEADLINE_FILE holds a Unix timestamp. It is re-read every minute, so the deadline can be moved
(e.g. extended once a run is known to be healthy) by rewriting the file. Exits when the pod is
gone or has been deleted.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from prime_cli.api.pods import PodsClient
from prime_cli.client import APIClient


def main():
    pod_id, deadline_file = sys.argv[1], Path(sys.argv[2])
    pods = PodsClient(APIClient())
    while True:
        deadline = float(deadline_file.read_text().strip())
        if time.time() >= deadline:
            print(f"{time.strftime('%m-%d %H:%M')} deadline reached: deleting {pod_id}", flush=True)
            pods.delete(pod_id)
            print("deleted", flush=True)
            return
        try:
            alive = [p for p in pods.list().data if p.id == pod_id]
            if not alive:
                print(f"{time.strftime('%m-%d %H:%M')} pod {pod_id} is gone; exiting", flush=True)
                return
        except Exception as e:  # transient API errors: keep the guard running
            print(f"{time.strftime('%m-%d %H:%M')} list failed: {e}", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
