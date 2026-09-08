import urllib.request
import json
import os

def check(url, method='GET', data=None):
    try:
        req = urllib.request.Request(url, method=method)
        if data:
            req.add_header('Content-Type', 'application/json')
            payload = json.dumps(data).encode('utf-8')
        else:
            payload = None
        with urllib.request.urlopen(req, data=payload, timeout=5) as r:
            body = r.read().decode('utf-8')
            ct = r.headers.get('Content-Type', '')
            parsed = json.loads(body) if 'json' in ct else len(body)
            print(f"[PASS] {method} {url} -> HTTP {r.status}")
            return True, parsed
    except Exception as e:
        print(f"[FAIL] {method} {url} -> {e}")
        return False, str(e)

print("=== 1. VERIFYING LOCALHOST APIS ===")
check("http://127.0.0.1:8000/health")
check("http://127.0.0.1:8000/vedas/status")
check("http://127.0.0.1:8000/vedas/satellite-feed?latitude=27.3389&longitude=88.6065")

ok, rt = check("http://127.0.0.1:8000/landslides/realtime")
if ok:
    print(f"   Total Real-time Landslide Events: {rt.get('total_active_events')}")
    first = rt['records'][0]
    print(f"   Sample GPS Telemetry: {first['name']} (Lat: {first['latitude']} N, Lon: {first['longitude']} E)")

ok_reg, reg = check("http://127.0.0.1:8000/regions/summary")
if ok_reg:
    print(f"   All 8 NER Regions: {list(reg.get('regions', {}).keys())}")

check("http://127.0.0.1:8000/predict/slope", method='POST', data={
    'latitude': 27.3389,
    'longitude': 88.6065,
    'slope_deg': 38.5,
    'current_pore_pressure_kpa': 42.8,
    'rainfall_24h_mm': 142.6
})

check("http://127.0.0.1:3000/")

print("\n=== 2. VERIFYING LOCALIZATION FILES ===")
locales_dir = "frontend/locales"
for f in ["en.json", "as.json", "hi.json", "bn.json", "bodo.json", "khasi.json"]:
    path = os.path.join(locales_dir, f)
    if os.path.exists(path):
        data = json.load(open(path, "r", encoding="utf-8"))
        print(f"[PASS] {f}: Valid JSON ({len(data)} root keys: {', '.join(data.keys())})")
    else:
        print(f"[FAIL] {f}: Missing file")

print("\n=== 3. VERIFYING GITHUB ACTION WORKFLOW ===")
workflow_path = ".github/workflows/i18n-translate.yml"
if os.path.exists(workflow_path):
    wf_content = open(workflow_path, "r", encoding="utf-8").read()
    if "taahamahdi/i18n-ai-translate@v5.3.0" in wf_content:
        print(f"[PASS] {workflow_path}: taahamahdi/i18n-ai-translate@v5.3.0 correctly configured")
