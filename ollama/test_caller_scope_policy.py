"""Shared consumers retain device clients and never authorize on unknown reads."""
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI, Request

from . import server


class CallerScopes(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, replies, alternatives=("group:services",), repeat=False):
        seen = []

        def verify(request):
            seen.append(request.url.params.get("scope"))
            self.assertEqual(request.headers["Authorization"], "Bearer synthetic-caller")
            value = replies[len(seen) - 1]
            if isinstance(value, Exception):
                raise value
            return httpx.Response(value, json={"valid": value == 200,
                "owner_id": "user:71", "username": "caller", "role": "agent"})

        app = FastAPI()
        app.add_middleware(server.SamcloudAuthMiddleware,
            verify_url="https://plane.test/auth/verify",
            required_scope="device:obelisk-services", alternate_scopes=alternatives)

        @app.get("/probe")
        async def probe(request: Request):
            return {"owner": request.state.caller["owner_id"]}

        real_client = httpx.AsyncClient
        async with real_client(transport=httpx.ASGITransport(app),
                base_url="https://provider.test") as caller:
            with patch.object(server, "AUTH_ENABLED", True), patch.object(
                    server.httpx, "AsyncClient", side_effect=lambda **kwargs:
                    real_client(transport=httpx.MockTransport(verify), **kwargs)):
                response = await caller.get("/probe", headers={
                    "Authorization": "Bearer synthetic-caller"})
                if repeat:
                    again = await caller.get("/probe", headers={
                        "Authorization": "Bearer synthetic-caller"})
                    self.assertEqual(again.status_code, response.status_code)
        return response, seen

    async def test_existing_device_caller_needs_no_group_or_second_read(self):
        response, seen = await self.exercise([200], repeat=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"owner": "user:71"})
        self.assertEqual(seen, ["device:obelisk-services"])

    async def test_shared_caller_after_confirmed_device_denial_is_cached(self):
        response, seen = await self.exercise([403, 200], repeat=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(seen, ["device:obelisk-services", "group:services"])

    async def test_unrelated_caller_is_denied_after_both_scope_checks(self):
        response, seen = await self.exercise([403, 403])
        self.assertEqual(response.status_code, 403)
        self.assertEqual(len(seen), 2)

    async def test_no_configured_alternative_preserves_original_policy(self):
        response, seen = await self.exercise([403], alternatives=())
        self.assertEqual(response.status_code, 403)
        self.assertEqual(seen, ["device:obelisk-services"])

    async def test_unknown_or_invalid_auth_never_tries_an_alternative(self):
        for reply, expected in ((401, 401), (500, 502),
                (httpx.ConnectError("offline"), 502)):
            with self.subTest(reply=repr(reply)):
                response, seen = await self.exercise([reply])
                self.assertEqual(response.status_code, expected)
                self.assertEqual(seen, ["device:obelisk-services"])

    def test_bad_alternative_or_empty_primary_refuses_at_startup(self):
        for primary, other in (("", "group:services"),
                ("device:obelisk-services", "group:services&scope=*")):
            with self.subTest(primary=primary, other=other):
                with self.assertRaises(ValueError):
                    server.SamcloudAuthMiddleware(FastAPI(), "https://plane.test",
                        primary, alternate_scopes=(other,))


if __name__ == "__main__":
    unittest.main()
