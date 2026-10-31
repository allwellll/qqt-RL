"""Local-only real WASM worker check; HTTP model bytes mapped from frozen exports."""
import http.server
import threading
import subprocess
import os
import json
from pathlib import Path
root=Path('../cx_publish_r1_1805_20261030/web').resolve()
f=Path('runs/v2_fresh_selection_20261031/R3_BROWSER_FIXTURES.json')
d=json.loads(f.read_text());assets={k:Path(v).resolve() for k,v in d['files'].items()}
class Handler(http.server.SimpleHTTPRequestHandler):
 def __init__(self,*a,**kw):super().__init__(*a,directory=str(root),**kw)
 def do_GET(self):
  key=self.path.split('?')[0]
  if key=='/fixture.html':raw=b'<script src="sim-bun-native-v2.js"></script><script src="r3_model_inputs.js"></script><script src="model_loader.js"></script><script src="model_worker.js"></script>';kind='text/html'
  elif key=='/fixtures.json':raw=f.read_bytes();kind='application/json'
  elif key.startswith('/models/') and key.split('/')[-1] in assets:raw=assets[key.split('/')[-1]].read_bytes();kind='application/octet-stream'
  else:return super().do_GET()
  self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 def log_message(self,*a):pass
s=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
threading.Thread(target=s.serve_forever,daemon=True).start()
e=dict(os.environ,WEB_URL=f'http://127.0.0.1:{s.server_port}/',LD_LIBRARY_PATH='/tmp/qqt-browser-libs/root/usr/lib/x86_64-linux-gnu')
try:
 with Path('runs/v2_fresh_selection_20261031/R3_BROWSER_ABI.log').open('a') as log:subprocess.run(['node','scripts/verify_v2_r3_browser_20261031.js'],env=e,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
finally:s.shutdown()
print('R3 browser worker ABI PASS')
