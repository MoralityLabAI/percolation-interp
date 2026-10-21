"""Rent, inspect and delete a Prime Intellect pod (e05b). Run with the prime CLI venv's python.

  python cloud/prime_pod.py offers [GPU_TYPE] [MAX_PRICE]     list single-GPU offers, cheapest first
  python cloud/prime_pod.py create OFFER_ID [NAME]            rent that offer with the registered SSH key
  python cloud/prime_pod.py status POD_ID                     status, IP and SSH line
  python cloud/prime_pod.py list                              this account's pods
  python cloud/prime_pod.py delete POD_ID                     stop billing

The CLI's own `pods create` cannot choose an SSH key; the API accepts pod.sshKeyId. The key
(~/.ssh/id_ed25519_prime) is registered on the account under the id below.
"""

from __future__ import annotations

import json
import sys

from prime_cli.api.availability import AvailabilityClient
from prime_cli.api.pods import PodsClient
from prime_cli.client import APIClient

SSH_KEY_ID = "3cc3fb95c186452f89cfc938ad309f5d"
PREFERRED_IMAGE = "ubuntu_22_cuda_12"


def offers(gpu_type: str = "A6000_48GB", max_price: float = 1.0):
    av = AvailabilityClient(APIClient()).get(gpu_type=gpu_type, gpu_count=1)
    rows = [g for gs in av.values() for g in gs if g.gpu_count == 1 and g.prices.price is not None
            and g.prices.price <= max_price]
    return sorted(rows, key=lambda g: g.prices.price)


def short(g) -> dict:
    return {"id": g.cloud_id[:6], "cloud_id": g.cloud_id, "provider": g.provider, "country": g.country,
            "price": g.prices.price, "stock": g.stock_status, "gpu_mem": g.gpu_memory, "spot": g.is_spot,
            "disk": g.disk.model_dump(), "vcpu": g.vcpu.model_dump(), "memory": g.memory.model_dump(),
            "images": g.images}


def create(offer_id: str, name: str = "e05b", gpu_type: str = "A6000_48GB"):
    match = [g for g in offers(gpu_type, 10.0) if g.cloud_id.startswith(offer_id)]
    assert len(match) == 1, f"{len(match)} offers match {offer_id}"
    g = match[0]
    image = PREFERRED_IMAGE if g.images and PREFERRED_IMAGE in g.images else (g.images or [None])[0]
    cfg = {
        "pod": {"name": name, "cloudId": g.cloud_id, "gpuType": g.gpu_type, "socket": g.socket, "gpuCount": 1,
                "diskSize": max(g.disk.default_count or 0, 80) if hasattr(g.disk, "default_count") else None,
                "vcpus": getattr(g.vcpu, "default_count", None), "memory": getattr(g.memory, "default_count", None),
                "image": image, "dataCenterId": g.data_center, "maxPrice": None, "country": None, "security": None,
                "jupyterPassword": None, "autoRestart": False, "customTemplateId": None, "envVars": None,
                "sshKeyId": SSH_KEY_ID},
        "provider": {"type": g.provider} if g.provider else {},
        "disks": None, "team": None,
    }
    pod = PodsClient(APIClient()).create(cfg)
    print(json.dumps({"pod_id": pod.id, "price": g.prices.price, "image": image, "provider": g.provider}))


def main():
    cmd, *args = sys.argv[1:]
    pods = PodsClient(APIClient())
    if cmd == "offers":
        for g in offers(*(args[:1] or ["A6000_48GB"]), *([float(args[1])] if len(args) > 1 else [])):
            print(json.dumps(short(g)))
    elif cmd == "create":
        create(*args)
    elif cmd == "status":
        for s in pods.get_status([args[0]]):
            print(json.dumps(s.model_dump(), default=str))
    elif cmd == "list":
        for p in pods.list().data:
            print(json.dumps({"id": p.id, "name": p.name, "status": p.status, "gpu": p.gpu_name if hasattr(p, "gpu_name")
                              else None}, default=str))
    elif cmd == "delete":
        pods.delete(args[0])
        print(f"deleted {args[0]}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
