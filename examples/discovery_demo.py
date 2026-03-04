import json
from pathlib import Path

from dataflux.discovery import scan_module

# 1. Path to our basic pipeline script
pipeline_script = Path(__file__).parent / "basic_pipeline.py"

# 2. Scan the module for callables
print(f"--- Scanning Module: {pipeline_script.name} ---")
schemas = scan_module(pipeline_script)

# 3. Print the results as formatted JSON
# This is exactly what FluxStudio will see
print(json.dumps(schemas, indent=2))

# 4. Verification
found_ops = [s["name"] for s in schemas]
assert "multiply" in found_ops
assert "add_noise" in found_ops
print("\nDiscovery Engine Verified!")
