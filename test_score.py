import urllib.request, json
url = "https://api.sofascore.com/api/v1/sport/football/scheduled-events/2026-04-15"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
evs = json.loads(urllib.request.urlopen(req).read()).get("events", [])
for ev in evs:
    h = ev["homeTeam"]["name"]
    if "Barcelona" in h or "Liverpool" in h:
        print(ev["homeTeam"]["name"], ev["homeScore"])
        print(ev["awayTeam"]["name"], ev["awayScore"])
        break
