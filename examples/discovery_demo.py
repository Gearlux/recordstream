"""Passive introspection: scan a module and get a JSON schema per callable (no manual tool defs).

Standalone, zero-arg, exit 0.
"""

import json

from sampleflux.discovery import scan_module


def main() -> None:
    module = "sampleflux.ops.numpy"
    print(f"--- Scanning Module: {module} ---")
    schemas = scan_module(module)

    print(json.dumps(schemas, indent=2))

    found = [s["name"] for s in schemas]
    assert "Threshold" in found
    assert "ConnectedComponents" in found
    print("\nDiscovery Engine Verified!")


if __name__ == "__main__":
    main()
