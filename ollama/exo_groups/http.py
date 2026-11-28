"""Bounded JSON metadata transport; no provider compute or retry policy."""

import json
import math


async def bounded_json(client, method, path, headers, *, payload=None):
    async with client.stream(method, path, headers=headers, json=payload) as response:
        return await bounded_response(response)


async def bounded_response(response):
    if (response.status_code != 200 or
            response.headers.get('content-type', '').split(';')[0] != 'application/json'):
        raise ValueError('plane HTTP metadata unavailable')
    raw = bytearray()
    async for chunk in response.aiter_bytes():
        if len(raw) + len(chunk) > 128 * 1024:
            raise ValueError('plane metadata exceeds bound')
        raw.extend(chunk)
    return decode_json(raw)


def decode_json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate plane field')
            value[key] = item
        return value

    def nonfinite(_value):
        raise ValueError('nonfinite plane field')

    def finite_float(raw):
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError('nonfinite plane field')
        return value

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite,
                       parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError('plane metadata is not an object')
    return value


def credential_headers(header):
    value = header()
    if (not isinstance(value, str) or not value.startswith('Bearer ')
            or len(value) <= 7 or any(ord(char) < 32 or ord(char) > 126 for char in value)):
        raise ValueError('coordinator credential unavailable')
    return {'Authorization': value, 'Accept': 'application/json', 'Cache-Control': 'no-cache'}
