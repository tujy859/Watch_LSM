"""Keep DREAMT transfers off proxies and bound to the selected interface."""
import os
import socket
import unittest
from unittest.mock import MagicMock, patch

import requests
from requests.adapters import HTTPAdapter

from scripts.download_dreamt_direct import DirectAdapter, DirectSession, IP_UNICAST_IF, authenticate, probe_direct, validate_manifest


class DirectDownloadTests(unittest.TestCase):
    def test_system_proxy_settings_are_ignored(self):
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://127.0.0.1:7897"}):
            with DirectSession("192.0.2.7", 17) as session:
                settings = session.merge_environment_settings("https://physionet.org/", {}, True, True, None)
                self.assertFalse(session.trust_env)
                self.assertFalse(settings["proxies"])

    def test_upstream_retry_mount_preserves_physical_interface(self):
        with DirectSession("192.0.2.7", 17) as session:
            session.mount("https://", HTTPAdapter(max_retries=5))
            adapter = session.get_adapter("https://physionet.org/")
            self.assertIsInstance(adapter, DirectAdapter)
            self.assertEqual(adapter.max_retries.total, 5)
            options = adapter.poolmanager.connection_pool_kw
            self.assertEqual(options["source_address"], ("192.0.2.7", 0))
            self.assertIn((socket.IPPROTO_IP, IP_UNICAST_IF, socket.htonl(17)), options["socket_options"])

    def test_cross_origin_and_proxy_fallback_are_rejected_before_network(self):
        with DirectSession("192.0.2.7", 17) as session:
            for url in ("https://physionet.org.evil.test/", "http://physionet.org/"):
                with self.assertRaises(ValueError):
                    session.send(requests.Request("GET", url).prepare())
            with self.assertRaises(ValueError):
                session.send(requests.Request("GET", "https://physionet.org/").prepare(), proxies={"https": "http://127.0.0.1:7897"})

    def test_manifest_prioritizes_64hz_without_omitting_other_files(self):
        files = [("data_256Hz/S1.csv", "a" * 64), ("data_64Hz/S1.csv", "b" * 64)]
        ordered = validate_manifest(files)
        self.assertEqual(ordered[0], files[1])
        self.assertEqual(set(ordered), set(files))

    def test_manifest_cannot_escape_destination(self):
        for name in ("../outside", "/absolute", "C:/outside", "dir\\outside"):
            with self.assertRaises(ValueError):
                validate_manifest([(name, "a" * 64)])

    def response(self, interface):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = 200
        transport = response.raw._connection.sock
        transport.getsockname.return_value = ("192.0.2.7", 45678)
        transport.getpeername.return_value = ("18.25.8.254", 443)
        transport.getsockopt.return_value = interface
        session = MagicMock()
        session.get.return_value = response
        return session

    def test_windows_get_option_is_already_host_order(self):
        result = probe_direct(self.response(17), "192.0.2.7", 17)
        self.assertEqual(result["interface_index"], 17)
        self.assertFalse(result["vpn_fallback_allowed"])

    def test_unexpected_tunnel_interface_is_not_accepted(self):
        with self.assertRaises(RuntimeError):
            probe_direct(self.response(41), "192.0.2.7", 17)

    def test_basic_access_does_not_require_a_website_login(self):
        session = MagicMock()
        session.get.return_value.__enter__.return_value.status_code = 200
        self.assertEqual(authenticate(session, "test-user", "test-password", "2.2.0", MagicMock()), "basic")
        session.post.assert_not_called()

    def login_session(self, cookie):
        session = MagicMock()
        basic, login, private = MagicMock(), MagicMock(), MagicMock()
        basic.__enter__.return_value.status_code = 403
        login.text = '<input name="csrfmiddlewaretoken" value="test-csrf">'
        private.__enter__.return_value.status_code = 403
        session.get.side_effect = [basic, login, private]
        session.cookies = {"sessionid": "test-session"} if cookie else {}
        return session

    def test_failed_web_login_is_not_reported_as_granted_access(self):
        save = MagicMock()
        with self.assertRaises(PermissionError):
            authenticate(self.login_session(False), "test-user", "test-password", "2.2.0", save)
        save.assert_any_call("login_not_confirmed", web_login_confirmed=False)

    def test_login_cookie_does_not_bypass_project_permission(self):
        save = MagicMock()
        with patch("builtins.input", return_value="q"), self.assertRaises(PermissionError):
            authenticate(self.login_session(True), "test-user", "test-password", "2.2.0", save)
        save.assert_any_call("waiting_for_project_access", cookie_manifest_status=403)
