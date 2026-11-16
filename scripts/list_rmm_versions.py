import json
import urllib.request

with urllib.request.urlopen("https://pypi.org/pypi/rmm-cu13/json") as resp:
    data = json.load(resp)
versions = list(data["releases"])
print("\n".join(versions))
