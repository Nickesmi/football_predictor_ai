import requests
try:
    r = requests.get("http://127.0.0.1:8000/api/health")
    print(r.status_code, r.text)
except Exception as e:
    print(e)
