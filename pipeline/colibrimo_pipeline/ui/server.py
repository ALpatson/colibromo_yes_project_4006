"""HTTP layer of ``cpipe ui`` (standard library only).

Listens on 127.0.0.1 only: other devices on the network cannot connect. Requests
must also name this computer in their ``Host`` (and ``Origin``, when sent)
header, so a web page from elsewhere cannot drive the tool through the browser
(DNS rebinding / cross-site requests).

Routes::

    GET  /                       the page (page.html)
    GET  /api/state?room=NAME    rooms, running run, the room's videos and progress
    PUT  /api/upload?room=&file= one video, raw body (streamed to data/uploads/<room>/)
    POST /api/start              {"room", "preset"}
    POST /api/resume             {"room"}
    POST /api/label              {"room", "file", "label": "bottom"|"middle"|"up"|""}
    POST /api/open-viewer        {"room"} -> {"url"}
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from colibrimo_pipeline.errors import PipelineError
from colibrimo_pipeline.ui.jobs import (
    PRESETS,
    Busy,
    Runner,
    UserError,
    clean_room_name,
    clean_video_filename,
    copy_to_viewer,
    list_rooms,
    list_videos,
    read_labels,
    read_progress,
    save_upload,
    set_label,
    video_details,
    viewer_running,
)

PAGE = Path(__file__).with_name("page.html")
HOST = "127.0.0.1"
MAX_UPLOAD_BYTES = 8 * 1024**3  # 8 GB per video
MAX_DRAIN_BYTES = (
    64 * 1024**2
)  # refused uploads up to this size are read before replying


class NotFound(Exception):
    """Unknown route (answered with 404)."""


class App:
    """State shared by all requests."""

    def __init__(self, runner: Runner, viewer_dir: Path | None, port: int):
        self.runner = runner
        self.viewer_dir = viewer_dir
        self.allowed_hosts = {f"{HOST}:{port}", f"localhost:{port}"}
        self._viewer_checked = (0.0, False)

    def viewer_running(self) -> bool:
        """Is the viewer's dev server up? Cached 10 s (the page polls every 2 s)."""
        checked_at, running = self._viewer_checked
        if time.monotonic() - checked_at > 10:
            running = viewer_running()
            self._viewer_checked = (time.monotonic(), running)
        return running

    def state(self, room: str | None) -> dict[str, Any]:
        runner = self.runner
        data: dict[str, Any] = {
            "running": runner.running,
            "running_room": runner.room if runner.running else None,
            "rooms": list_rooms(runner.data_dir),
            "presets": PRESETS,
            "trainer": runner.overrides.get("train", {}).get("trainer", "gsplat"),
            "viewer_dir": str(self.viewer_dir) if self.viewer_dir else None,
        }
        if room:
            room = clean_room_name(room)
            labels = read_labels(runner.uploads_dir, room)
            videos = [
                {
                    "name": v.name,
                    "size": v.stat().st_size,
                    "label": labels.get(v.name, ""),
                    "details": video_details(v),
                }
                for v in list_videos(runner.uploads_dir, room)
            ]
            orientations = {v["details"]["orientation"] for v in videos if v["details"]}
            data["room"] = {
                "name": room,
                "videos": videos,
                "mixed_orientation": len(orientations) > 1,
                "missing_middle": bool(videos) and "middle" not in labels.values(),
                "progress": read_progress(runner.run_dir(room)),
                "error": runner.errors.get(room),
            }
        return data


class Handler(BaseHTTPRequestHandler):
    server_version = "cpipe-ui"
    app: App  # set by make_server

    # ---- helpers ----

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (stdlib name)
        pass  # keep the console for pipeline output

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    def _error(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        self._json({"error": message}, status)

    def _query(self) -> dict[str, str]:
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

    def _body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 1 << 16:
            raise UserError("Request too large.")
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            raise UserError("Invalid JSON.") from exc
        if not isinstance(data, dict):
            raise UserError("Expected a JSON object.")
        return data

    def _allowed(self) -> bool:
        """Only requests addressed to this computer, from this tool's own page."""
        if self.headers.get("Host") not in self.app.allowed_hosts:
            return False
        origin = self.headers.get("Origin")
        return origin is None or urlparse(origin).netloc in self.app.allowed_hosts

    def _handle(self, action) -> None:
        if not self._allowed():
            self._discard_body()
            self._error(
                "Forbidden: this tool only accepts local requests.",
                HTTPStatus.FORBIDDEN,
            )
            return
        try:
            action()
        except NotFound:
            self._error("Not found.", HTTPStatus.NOT_FOUND)
        except Busy as exc:
            self._error(str(exc), HTTPStatus.CONFLICT)
        except PipelineError as exc:  # UserError, ConfigError, ...
            self._error(str(exc))

    # ---- routes ----

    def do_GET(self) -> None:  # noqa: N802 (stdlib name)
        def action():
            route = urlparse(self.path).path
            if route == "/":
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif route == "/api/state":
                state = self.app.state(self._query().get("room"))
                state["viewer_running"] = self.app.viewer_running()
                self._json(state)
            else:
                self._error("Not found.", HTTPStatus.NOT_FOUND)

        self._handle(action)

    def _discard_body(self) -> None:
        """Read (and drop) an unread request body before answering an error.

        Otherwise the client is still sending when the reply comes and the
        connection is reset (on Windows), so the browser never sees the error
        message. Very large bodies are not drained: the connection is closed.
        """
        length = int(self.headers.get("Content-Length") or 0)
        self.close_connection = True
        if 0 < length <= MAX_DRAIN_BYTES:
            while length > 0:
                chunk = self.rfile.read(min(1 << 20, length))
                if not chunk:
                    break
                length -= len(chunk)

    def do_PUT(self) -> None:  # noqa: N802
        def action():
            body_read = False
            try:
                if urlparse(self.path).path != "/api/upload":
                    raise NotFound
                query = self._query()
                length = int(self.headers.get("Content-Length") or -1)
                if length < 0 or length > MAX_UPLOAD_BYTES:
                    raise UserError("Missing or too large Content-Length.")
                # Check the names before reading anything.
                clean_room_name(query.get("room", ""))
                clean_video_filename(query.get("file", ""))
                body_read = True
                path = save_upload(
                    self.app.runner.uploads_dir,
                    query.get("room", ""),
                    query.get("file", ""),
                    self.rfile,
                    length,
                )
            except (NotFound, PipelineError):
                if not body_read:
                    self._discard_body()
                raise
            self._json(
                {"saved": path.name, "size": path.stat().st_size}, HTTPStatus.CREATED
            )

        self._handle(action)

    def do_POST(self) -> None:  # noqa: N802
        def action():
            route = urlparse(self.path).path
            body = self._body_json()
            runner = self.app.runner
            if route == "/api/start":
                runner.start(
                    str(body.get("room", "")), str(body.get("preset", "balanced"))
                )
                self._json({"started": True})
            elif route == "/api/resume":
                runner.resume(str(body.get("room", "")))
                self._json({"started": True})
            elif route == "/api/label":
                labels = set_label(
                    runner.uploads_dir,
                    str(body.get("room", "")),
                    str(body.get("file", "")),
                    str(body.get("label", "")),
                )
                self._json({"labels": labels})
            elif route == "/api/open-viewer":
                if self.app.viewer_dir is None:
                    raise UserError(
                        "No viewer folder configured (cpipe ui --viewer DIR)."
                    )
                room = clean_room_name(str(body.get("room", "")))
                url = copy_to_viewer(runner.run_dir(room), self.app.viewer_dir, room)
                self._json({"url": url, "viewer_running": viewer_running()})
            else:
                self._error("Not found.", HTTPStatus.NOT_FOUND)

        self._handle(action)


def make_server(app: App, port: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app})
    return ThreadingHTTPServer((HOST, port), handler)


def serve(
    data_dir: Path,
    overrides: dict[str, Any],
    viewer_dir: Path | None,
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Run the UI until Ctrl+C."""
    app = App(Runner(data_dir, overrides), viewer_dir, port)
    server = make_server(app, port)
    url = f"http://{HOST}:{port}/"
    print(f"cpipe ui: {url} (this computer only). Data: {Path(data_dir).resolve()}")
    print(
        f"Viewer folder: {viewer_dir or 'not configured (--viewer)'}. Ctrl+C to stop."
    )
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
