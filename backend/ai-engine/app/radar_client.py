"""
Doppler Weather Radar Client
Ingests live IMD / RainViewer Doppler radar tile frames for India and the North Eastern Region.
Provides XYZ tile templates and animated past/nowcast timestamps.
"""

import urllib.request
import json
import time
from typing import Dict, Any, List

class RadarClient:
    def __init__(self):
        self.cache = None
        self.cache_time = 0
        self.ttl = 300  # 5 minutes cache

    def get_live_radar_frames(self) -> Dict[str, Any]:
        now = time.time()
        if self.cache and (now - self.cache_time < self.ttl):
            return self.cache

        url = "https://api.rainviewer.com/public/weather-maps.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MDoNER-Landslide-EWS/2.4"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                host = data.get("host", "https://tilecache.rainviewer.com")
                radar_info = data.get("radar", {})
                past_frames = radar_info.get("past", [])
                nowcast_frames = radar_info.get("nowcast", [])

                # Standard tile URL format: {host}{path}/256/{z}/{x}/{y}/2/1_1.png
                frames_list = []
                for item in past_frames:
                    t_val = item.get("time")
                    path_val = item.get("path")
                    frames_list.append({
                        "time": t_val,
                        "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_val)),
                        "path": path_val,
                        "tile_url_template": f"{host}{path_val}/256/{{z}}/{{x}}/{{y}}/2/1_1.png",
                        "is_nowcast": False
                    })

                for item in nowcast_frames:
                    t_val = item.get("time")
                    path_val = item.get("path")
                    frames_list.append({
                        "time": t_val,
                        "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_val)),
                        "path": path_val,
                        "tile_url_template": f"{host}{path_val}/256/{{z}}/{{x}}/{{y}}/2/1_1.png",
                        "is_nowcast": True
                    })

                latest = frames_list[-1] if frames_list else None

                result = {
                    "status": "SUCCESS",
                    "host": host,
                    "generated": data.get("generated"),
                    "total_frames": len(frames_list),
                    "frames": frames_list,
                    "latest_frame": latest,
                    "source": "IMD Doppler Radar Network via RainViewer Maps API",
                    "coverage": "All-India & Himalayan Doppler Radar Stations (Cherrapunji, Mohanbari, Agartala, Kolkata, Patna)",
                    "zoom_earth_url": "https://zoom.earth/maps/radar/#overlays=radar,wind"
                }

                self.cache = result
                self.cache_time = now
                return result

        except Exception as e:
            print(f"[RadarClient] Error: {e}")
            return {
                "status": "FALLBACK",
                "host": "https://tilecache.rainviewer.com",
                "total_frames": 0,
                "frames": [],
                "latest_frame": None,
                "source": "IMD Doppler Radar",
                "zoom_earth_url": "https://zoom.earth/maps/radar/#overlays=radar,wind"
            }

radar_client = RadarClient()
