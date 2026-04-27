import requests
import json

r = requests.get(
    "https://ffiec.cfpb.gov/v2/data-browser-api/view/filers",
    params={"name": "bank", "years": "2022"},
    timeout=30
)

data = r.json()
institutions = data.get("institutions", [])

# print all of them so we can pick
for inst in sorted(institutions, key=lambda x: x.get("count", 0), reverse=True)[:40]:
    print(f"{inst['count']:>10,}  {inst['lei']}  {inst['name']}")