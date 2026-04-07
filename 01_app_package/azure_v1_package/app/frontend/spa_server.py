import http.server
import socketserver
import os
import sys

PORT = 5173
DIRECTORY = "/Users/integrationwings/Desktop/LLM_Wrap/oracle-fusion-slm/apps/iwerp-ui/dist"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        # If the path doesn't exist as a file, serve index.html (SPA fallback)
        path = self.translate_path(self.path)
        if not os.path.exists(path):
            self.path = "/index.html"
        return super().do_GET()

if __name__ == "__main__":
    os.chdir(DIRECTORY)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"IWERP SPA Server serving at port {PORT}")
        httpd.serve_forever()
