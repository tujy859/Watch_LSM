"""Download DREAMT using the official client over a bound Windows WLAN socket.

No system routes or VPN settings are changed. Proxy use and cross-origin
redirects are rejected. Credentials are prompted locally and stay in memory.
"""
import argparse
from datetime import datetime, timezone
import getpass
from html.parser import HTMLParser
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import subprocess
import sys
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection

ROOT = Path(__file__).resolve().parents[1]
CLIENT_SHA = "54ebd21308c368a9d1e8cfd8a673b111cacead0e"
ORIGIN = "https://physionet.org"
IP_UNICAST_IF = getattr(socket, "IP_UNICAST_IF", 31)


class LoginTokenParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.token = None

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        if tag == "input" and values.get("name") == "csrfmiddlewaretoken":
            self.token = values.get("value")


def authenticate(session, username, password, version, save):
    """Check ordinary Basic auth, then diagnose via a normal CSRF-protected login.

    Cookies, if needed by the site, remain only in this process. This never signs
    a data agreement and never treats a successful login as dataset permission.
    """
    url = ORIGIN + f"/files/dreamt/{version}/SHA256SUMS.txt"
    session.auth = requests.auth.HTTPBasicAuth(username, password)
    with session.get(url, timeout=(15, 30)) as response:
        status = response.status_code
    save("checking_credentials", basic_manifest_status=status)
    if status == 200:
        return "basic"
    if status not in (401, 403):
        raise RuntimeError(f"PhysioNet file endpoint returned HTTP {status}")
    session.auth = None
    login = ORIGIN + "/login/"
    page = session.get(login, timeout=(15, 30))
    page.raise_for_status()
    parser = LoginTokenParser()
    parser.feed(page.text)
    if not parser.token:
        raise RuntimeError("Login page requires browser verification; use the PhysioNet website")
    response = session.post(login, data={"username": username, "password": password, "csrfmiddlewaretoken": parser.token, "next": "/"}, headers={"Referer": login, "Origin": ORIGIN}, timeout=(15, 30))
    response.raise_for_status()
    if not session.cookies.get("sessionid"):
        save("login_not_confirmed", web_login_confirmed=False)
        raise PermissionError("Website login was not accepted. Check username/password, email activation, or additional login verification.")
    save("checking_project_access", web_login_confirmed=True)
    while True:
        with session.get(url, timeout=(15, 30)) as response:
            status = response.status_code
        if status == 200:
            save("project_access_verified", cookie_manifest_status=200)
            return "cookie"
        if status not in (401, 403):
            raise RuntimeError(f"Authenticated file endpoint returned HTTP {status}")
        save("waiting_for_project_access", cookie_manifest_status=status)
        print(f"Website login succeeded, but DREAMT {version} files return HTTP {status}.", flush=True)
        print(f"Open {ORIGIN}/content/dreamt/{version}/ and check the agreement/access status.", flush=True)
        answer = input("After access is granted, press Enter to retry using the same session; q to stop: ")
        if answer.strip().lower() == "q":
            raise PermissionError("Dataset access is not granted; user stopped the retry")


def validate_manifest(files):
    for name, digest in files:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or not path.parts:
            raise ValueError("Unsafe path in publisher manifest")
        if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError("Invalid publisher SHA-256")
    # Fetch benchmark-ready 64 Hz data first, while still downloading all files.
    return sorted(files, key=lambda item: (not item[0].startswith("data_64Hz/"), item[0]))


class DirectAdapter(HTTPAdapter):
    def __init__(self, source_ip, interface_index, **kwargs):
        self.source_ip, self.interface_index = source_ip, interface_index
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
        kwargs["source_address"] = (self.source_ip, 0)
        kwargs["socket_options"] = [*HTTPConnection.default_socket_options, (socket.IPPROTO_IP, IP_UNICAST_IF, socket.htonl(self.interface_index))]
        return super().init_poolmanager(connections, maxsize, block, **kwargs)


class DirectSession(requests.Session):
    def __init__(self, source_ip, interface_index):
        self.source_ip, self.interface_index = source_ip, interface_index
        super().__init__()
        self.trust_env = False
        self.proxies.clear()

    def mount(self, prefix, adapter):
        # The official downloader replaces retry adapters for each file. Retain
        # its retry policy without losing the explicit physical-interface bind.
        if not isinstance(adapter, DirectAdapter):
            adapter = DirectAdapter(self.source_ip, self.interface_index, max_retries=adapter.max_retries)
        return super().mount(prefix, adapter)

    def send(self, request, **kwargs):
        parsed = urlsplit(request.url)
        if parsed.scheme != "https" or parsed.hostname != "physionet.org" or parsed.port not in (None, 443):
            raise ValueError("Direct downloader allows only the PhysioNet HTTPS origin")
        if any(kwargs.get("proxies", {}).values()):
            raise ValueError("Proxy use is disabled for this download")
        return super().send(request, **kwargs)


def probe_direct(session, source_ip, interface_index):
    with session.get(ORIGIN + "/content/dreamt/2.2.0/", stream=True, timeout=(15, 30)) as response:
        response.raise_for_status()
        connection = response.raw._connection
        transport = connection.sock if connection else None
        if transport is None:
            raise RuntimeError("Cannot verify the download socket; refusing a proxy fallback")
        actual_ip = transport.getsockname()[0]
        # Windows sets this option in network order but returns it in host order.
        actual_index = transport.getsockopt(socket.IPPROTO_IP, IP_UNICAST_IF)
        if actual_ip != source_ip or actual_index != interface_index:
            raise RuntimeError("Socket is not bound to the requested physical interface")
        return {"local_ip": actual_ip, "interface_index": actual_index, "remote_ip": transport.getpeername()[0], "https_status": response.status_code, "system_proxy_ignored": True, "vpn_fallback_allowed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default="tujy859")
    parser.add_argument("--version", choices=["2.2.0"], default="2.2.0")
    parser.add_argument("--source-ip", required=True)
    parser.add_argument("--interface-index", type=int, required=True)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if os.name != "nt" or args.interface_index < 1:
        parser.error("A valid Windows physical-interface index is required")
    revision = subprocess.check_output(["git", "-C", str(ROOT / "repos/physionet-client"), "rev-parse", "HEAD"], text=True).strip()
    if revision != CLIENT_SHA:
        raise RuntimeError("Unexpected PhysioNet client revision")
    destination = ROOT / "datasets/ppg_bench_raw" / f"dreamt-{args.version}"
    state_path = ROOT / ".local/dreamt-download-status.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"version": args.version, "client_revision": revision, "destination": str(destination.relative_to(ROOT)), "phase": "checking_direct_connection", "completed_files": 0, "downloaded_bytes_this_run": 0}

    def save(phase, **fields):
        state.update(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **fields)
        temporary = state_path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(state_path)

    with DirectSession(args.source_ip, args.interface_index) as session:
        state["direct_connection"] = probe_direct(session, args.source_ip, args.interface_index)
    save("direct_connection_verified")
    print("Direct WLAN HTTPS connection verified. No VPN fallback.", flush=True)
    if args.probe_only:
        print(json.dumps(state["direct_connection"], indent=2))
        return 0
    if not sys.stdin.isatty():
        raise RuntimeError("Enter the password only in an interactive local terminal")
    save("waiting_for_password")
    password = getpass.getpass("PhysioNet password for " + args.username + ": ")
    if not password:
        save("cancelled")
        return 1
    module = importlib.import_module("physionet.download")
    client_type, manifest_function, download_function = module.PhysioNetClient, module._get_file_manifest, module._download_file
    verify_function = module._verify_checksum
    verified = set()
    credential_session = DirectSession(args.source_ip, args.interface_index)
    try:
        auth_mode = authenticate(credential_session, args.username, password, args.version, save)
    except Exception as error:
        credential_session.close()
        password = None
        save("authentication_stopped", error_type=type(error).__name__, error=str(error))
        print(error, flush=True)
        return 1
    state["auth_mode"] = auth_mode

    class Client(client_type):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            previous = self.session
            self.session = DirectSession(args.source_ip, args.interface_index)
            self.session.auth = previous.auth if auth_mode == "basic" else None
            self.session.cookies.update(credential_session.cookies)
            self.session.headers.update(previous.headers)
            previous.close()

    def manifest(*values):
        files = validate_manifest(manifest_function(*values))
        (state_path.parent / f"dreamt-{args.version}-manifest.json").write_text(json.dumps(files, indent=2))
        save("downloading", planned_files=len(files))
        return files

    def download(url, target, *values, **kwargs):
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Download path escaped its versioned directory")
        save("downloading", current_file=target.relative_to(destination).as_posix())
        size = download_function(url, target, *values, **kwargs)
        state["downloaded_files_this_run"] = state.get("downloaded_files_this_run", 0) + 1
        state["downloaded_bytes_this_run"] += size
        save("downloading")
        return size

    def verify(target, digest):
        valid = verify_function(target, digest)
        if valid:
            verified.add(target.relative_to(destination).as_posix())
            state["completed_files"] = len(verified)
            save("downloading")
        return valid

    module.PhysioNetClient, module._get_file_manifest, module._download_file, module._verify_checksum = Client, manifest, download, verify
    try:
        module.download("dreamt", version=args.version, output_dir=str(destination.parent), source="physionet", username=args.username, password=password, retries=5)
        (destination / "source.json").write_text(json.dumps({"dataset": "dreamt", "version": args.version, "source": ORIGIN + f"/content/dreamt/{args.version}/", "client_revision": revision, "direct_connection": state["direct_connection"], "complete": True}, indent=2))
        save("complete")
        return 0
    except Exception as error:
        save("failed", error_type=type(error).__name__, error=str(error))
        print("Download stopped:", error, flush=True)
        return 1
    finally:
        password = None
        credential_session.close()


if __name__ == "__main__":
    raise SystemExit(main())
