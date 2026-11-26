"""Optional, passive S8 protocol diagnostic. Does not feed or overwrite vision.

The gateway framing/route 3915 were verified against a live capture on 2026-11-26.
Type 1 is intentionally left unresolved. Protobuf schemas come from the supplied
game descriptor directory, not from guessing a message type from its contents.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path


SERVER_MESSAGES = {3915: 'S8DrawMapDataAck', 3922: 'S8DrawPushChangeNotify'}


def reassemble_tcp(segments):
    """Reassemble (sequence, timestamp, packet_number, data); reject data loss."""
    segments = sorted(segments)
    if not segments:
        return b'', []
    base = segments[0][0]
    data = bytearray()
    spans = []
    for seq, timestamp, packet, payload in segments:
        offset = seq - base
        if offset > len(data):
            raise ValueError(f'TCP gap at {len(data)}: next offset {offset}')
        overlap = max(0, len(data) - offset)
        common = min(overlap, len(payload))
        if data[offset:offset + common] != payload[:common]:
            raise ValueError(f'TCP retransmission conflict at {offset}')
        suffix = payload[overlap:]
        if suffix:
            spans.append({'start': len(data), 'end': len(data) + len(suffix),
                          'ts': timestamp, 'packet': packet})
            data.extend(suffix)
    return bytes(data), spans


def split_gateway_frames(data):
    offset = 0
    while offset < len(data):
        if len(data) - offset < 4:
            raise ValueError(f'incomplete gateway header at {offset}')
        kind = data[offset]
        length = int.from_bytes(data[offset + 1:offset + 4], 'big')
        if kind not in (0, 1, 2):
            raise ValueError(f'unknown gateway frame type {kind} at {offset}')
        if length > 8 * 1024 * 1024:
            raise ValueError(f'gateway frame too large at {offset}')
        end = offset + 4 + length
        if end > len(data):
            raise ValueError(f'incomplete gateway frame at {offset}')
        yield offset, kind, data[offset + 4:end]
        offset = end


class ZstdDecoder:
    """Use the already-installed Wireshark DLL, with bounded decompression."""
    def __init__(self, dll_path: Path):
        self.dll = ctypes.CDLL(str(dll_path.resolve()))
        self.dll.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.dll.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
        self.dll.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                           ctypes.c_void_p, ctypes.c_size_t]
        self.dll.ZSTD_decompress.restype = ctypes.c_size_t
        self.dll.ZSTD_isError.argtypes = [ctypes.c_size_t]
        self.dll.ZSTD_isError.restype = ctypes.c_uint

    def __call__(self, data: bytes) -> bytes:
        limit = 32 * 1024 * 1024
        size = self.dll.ZSTD_getFrameContentSize(data, len(data))
        if size == 2**64 - 1:  # ZSTD_CONTENTSIZE_UNKNOWN
            size = limit
        if size > limit:  # Includes ZSTD_CONTENTSIZE_ERROR.
            raise ValueError('invalid or oversized zstd frame')
        output = ctypes.create_string_buffer(max(1, size))
        result = self.dll.ZSTD_decompress(output, size, data, len(data))
        if self.dll.ZSTD_isError(result):
            raise ValueError('zstd decompression failed')
        return output.raw[:result]


class S8MessageDecoder:
    """Optional protobuf dependency is loaded only by the diagnostic command."""
    def __init__(self, descriptor_dir: Path):
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
        self.pool = descriptor_pool.DescriptorPool()
        self.descriptor_sha256 = {}
        self.pool.AddSerializedFile(descriptor_pb2.DESCRIPTOR.serialized_pb)
        loaded = {'google/protobuf/descriptor.proto'}
        visiting = set()

        def add(name):
            if name in loaded:
                return
            if name in visiting:
                raise ValueError(f'descriptor dependency cycle: {name}')
            visiting.add(name)
            path = descriptor_dir / Path(name).with_suffix('.pb').name
            content = path.read_bytes()
            self.descriptor_sha256[path.name] = hashlib.sha256(content).hexdigest()
            descriptors = descriptor_pb2.FileDescriptorSet.FromString(content)
            for descriptor in descriptors.file:
                if descriptor.name in loaded:
                    continue
                for dependency in descriptor.dependency:
                    add(dependency)
                self.pool.Add(descriptor)
                loaded.add(descriptor.name)
            visiting.remove(name)

        add('proto/gameplays8_api.proto')
        self.messages = {route: message_factory.GetMessageClass(
            self.pool.FindMessageTypeByName(f'rpc.{name}'))
            for route, name in SERVER_MESSAGES.items()}

    def decode_server_envelope(self, data: bytes):
        from google.protobuf import json_format
        if len(data) < 16:
            return None
        route = int.from_bytes(data[14:16], 'big')
        if route not in self.messages:
            return None
        message = self.messages[route].FromString(data[16:])
        body = json_format.MessageToDict(message, preserving_proto_field_name=True)
        if route == 3915:
            plots = body.get('plots', [])
            if len(plots) != 36 or {p.get('no') for p in plots} != set(range(1, 37)):
                raise ValueError('S8 snapshot does not contain 36 distinct plot numbers')
        return {'route': route, 'message': SERVER_MESSAGES[route], 'body': body,
                'brush_records': extract_brush_records(body)}


def extract_brush_records(body):
    """Preserve IDs/levels verbatim. Do not assume facadeLevel is a bonus."""
    records = []

    def item_record(item, plot=None):
        effect = item.get('task', {}).get('effect', {})
        records.append({
            'item_unique_id': item.get('uniqueId'), 'item_id': item.get('itemId'),
            'plot_no': plot.get('no') if plot else None,
            'raw_level': item.get('rawLevel', 0),
            'facade_level': plot.get('facadeLevel', 0) if plot else None,
            'facade_item_id': plot.get('facadeItemId', 0) if plot else None,
            'tint_effect_id': effect.get('effectId', 0),
            'brush_effect_id': effect.get('selectableEffectId', 0),
            'brush_effect_params': effect.get('selectableEffectParams', ''),
            'brush_option_ids': effect.get('selectableFromOptIds', []),
            'brush_library': effect.get('selectableEffectType'),
        })

    for key in ('plots', 'changePlots'):
        for plot in body.get(key, []):
            for item in plot.get('items', []):
                item_record(item, plot)
    for key in ('items', 'changeItems'):
        for item in body.get(key, []):
            item_record(item)
    return records


def compare_brush_ids(protocol_id, ocr_ids, *, paired: bool):
    """Caller must establish same paint/event before passing paired=True."""
    if protocol_id is None:
        return 'protocol_unavailable'
    if not paired:
        return 'unpaired'
    if protocol_id == 0:
        return 'protocol_no_brush'
    ids = set(ocr_ids)
    if not ids:
        return 'ocr_unknown'
    if protocol_id not in ids:
        return 'mismatch'
    return 'exact' if len(ids) == 1 else 'candidate_contains'
