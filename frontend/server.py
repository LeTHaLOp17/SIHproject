"""
Zero-Dependency Localhost Web Server for Landslide EWS Command Center
Runs on port 3000 and serves the interactive frontend dashboard.
"""

import http.server
import socketserver
import os
import sys

PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Enable CORS and caching headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 MDoNER AI LANDSLIDE EARLY WARNING & MONITORING SYSTEM")
    print(f"📡 Serving Web Command Center on: http://localhost:{PORT}")
    print(f"📂 Serving directory: {DIRECTORY}")
    print("=" * 70)
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            httpd.shutdown()
