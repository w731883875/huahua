"""Read a bounded, game-only pcapng and export verified S8 server snapshots.

This is a diagnostic sidecar, not an always-on replacement for screen recording.
Run with --help. Requires local Wireshark and optional Python protobuf.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huahua_assistant.protocol_probe import (
    S8MessageDecoder, ZstdDecoder, reassemble_tcp, split_gateway_frames,
)


def analyze_capture(pcap: Path, descriptor_dir: Path, wireshark: Path, server_port: int):
    fields = ['frame.number', 'frame.time_epoch', 'tcp.stream', 'ip.src',
              'tcp.srcport', 'tcp.seq_raw', 'tcp.payload']
    command = [str(wireshark / 'tshark.exe'), '-n', '-r', str(pcap),
               '-Y', f'tcp.len > 0 && tcp.port == {server_port}', '-T', 'fields']
    for field in fields:
        command.extend(['-e', field])
    result = subprocess.run(command, capture_output=True, check=True, timeout=60)
    groups = defaultdict(list)
    for line in result.stdout.decode('utf-8').splitlines():
        values = line.split('\t')
        if len(values) == len(fields) and values[-1]:
            frame, stamp, stream, address, port, sequence, payload = values
            groups[(stream, address, int(port))].append(
                (int(sequence), float(stamp), int(frame), bytes.fromhex(payload)))
    decoder = S8MessageDecoder(descriptor_dir)
    zstd = ZstdDecoder(wireshark / 'zstd.dll')
    report = {'schema': 'huahua-protocol-probe-v1', 'source': str(pcap.resolve()),
              'source_sha256': hashlib.sha256(pcap.read_bytes()).hexdigest(),
              'descriptor_sha256': decoder.descriptor_sha256,
              'mode': 'passive_diagnostic', 'streams': [], 'messages': [],
              'limitations': ['type1_not_decoded', 'no_automatic_ocr_event_pairing',
                              'historical_descriptors_require_version_validation']}
    for (stream_id, _, port), segments in groups.items():
        server = port == server_port
        stream_report = {'stream': stream_id, 'direction': 'server' if server else 'client',
                         'segments': len(segments), 'frame_types': {}, 'errors': []}
        report['streams'].append(stream_report)
        try:
            data, spans = reassemble_tcp(segments)
            frames = list(split_gateway_frames(data))
        except ValueError as error:
            stream_report['errors'].append(str(error))
            continue
        stream_report['frame_types'] = dict(Counter(kind for _, kind, _ in frames))
        for offset, kind, payload in frames:
            if not server or kind == 1:
                continue
            try:
                plain = zstd(payload) if kind == 2 else payload
                message = decoder.decode_server_envelope(plain)
                if message is None:
                    continue
                end = offset + 4 + len(payload)
                covering = [s for s in spans if s['start'] < end and s['end'] > offset]
                message.update({'stream': stream_id, 'offset': offset, 'frame_type': kind,
                                'ts': max(s['ts'] for s in covering),
                                'packet_numbers': [s['packet'] for s in covering]})
                report['messages'].append(message)
            except Exception as error:
                # Continue other frames, but retain a visible failure; never invent IDs.
                stream_report['errors'].append(f'offset {offset}: {type(error).__name__}: {error}')
    report['messages'].sort(key=lambda message: message['ts'])
    report['summary'] = {
        'decoded_s8_messages': len(report['messages']),
        'unresolved_type1_frames': sum(s['frame_types'].get(1, 0) for s in report['streams']),
        'nonzero_brush_records': sum(bool(item['brush_effect_id'])
                                    for m in report['messages'] for item in m['brush_records']),
        'errors': sum(len(s['errors']) for s in report['streams']),
        'comparison_status': 'unpaired',
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pcap', type=Path)
    parser.add_argument('--descriptor-dir', required=True, type=Path,
                        help='Directory containing gameplays8_api.pb and its dependencies')
    parser.add_argument('--wireshark-dir', type=Path, default=Path(r'C:\Program Files\Wireshark'))
    parser.add_argument('--server-port', required=True, type=int,
                        help='Server port verified as owned by the game process')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() == args.pcap.resolve():
        parser.error('Output must not overwrite the capture')
    report = analyze_capture(args.pcap, args.descriptor_dir, args.wireshark_dir, args.server_port)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
