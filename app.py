import base64
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, UnidentifiedImageError

from infer import estimate_depth

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8000"))
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
INDEX_FILE = Path(__file__).with_name("index.html")


class DepthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if urlsplit(self.path).path != "/":
            self.send_error(404)
            return

        try:
            page = INDEX_FILE.read_bytes()
        except OSError:
            logging.exception("Could not read the UI page")
            self.send_error(500, "Could not read index.html")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/depth":
            self.send_error(404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.send_json(400, {"error": "A valid image upload is required."})
            return

        if content_length <= 0:
            self.send_json(400, {"error": "Choose an image before generating depth."})
            return
        if content_length > MAX_UPLOAD_BYTES:
            self.send_json(413, {"error": "Images must be 20 MB or smaller."})
            return

        try:
            image = Image.open(BytesIO(self.rfile.read(content_length)))
            image.load()
            image = image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            self.send_json(400, {"error": "That file could not be opened as an image."})
            return

        try:
            grayscale, colored = estimate_depth(image)
            self.send_json(
                200,
                {
                    "grayscale": self.encode_png(grayscale),
                    "colored": self.encode_png(colored),
                },
            )
        except Exception:
            logging.exception("Depth estimation failed")
            self.send_json(
                500,
                {"error": "Depth estimation failed. Check the server terminal for details."},
            )

    def send_json(self, status: int, data: dict[str, str]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def encode_png(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def log_message(self, format: str, *args: object) -> None:
        logging.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    server = ThreadingHTTPServer((HOST, PORT), DepthHandler)
    print(f"Depth Studio is ready at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Depth Studio...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
