import urllib.request, json
url = "https://api.sofascore.com/api/v1/sport/football/scheduled-events/2026-04-15"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
evs = json.loads(urllib.request.urlopen(req).read()).get("events", [])
for ev in evs:
    h = ev["homeTeam"]["name"]
    a = ev["awayTeam"]["name"]
    if "Barcelona" in h or "Barcelona" in a:
        eid = ev["id"]
        print(f"Found match: {h} vs {a} (ID: {eid})")
        
        stat_url = f"https://api.sofascore.com/api/v1/event/{eid}/statistics"
        stat_req = urllib.request.Request(stat_url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            stats = json.loads(urllib.request.urlopen(stat_req).read())
            print("Statistics keys:", stats.keys())
            if stats.get("statistics"):
                for period in stats["statistics"]:
                    if period["period"] == "ALL":
                        for group in period["groups"]:
                            for item in group["statisticsItems"]:
                                print(f" - {item['name']}: {item['home']} vs {item['away']}")
        except Exception as e:
            print("Stat error:", e)
        break
