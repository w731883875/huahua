import unittest

from huahua_assistant.protocol_probe import (
    compare_brush_ids, extract_brush_records, reassemble_tcp, split_gateway_frames,
)


class ProtocolProbeTests(unittest.TestCase):
    def test_reassembles_out_of_order_and_retransmission(self):
        data, spans = reassemble_tcp([(103, 2.0, 2, b'def'), (100, 1.0, 1, b'abcd'),
                                      (100, 3.0, 3, b'abcd')])
        self.assertEqual(data, b'abcdef')
        self.assertEqual(spans[-1]['end'], 6)

    def test_missing_bytes_are_never_filled(self):
        with self.assertRaisesRegex(ValueError, 'gap'):
            reassemble_tcp([(100, 1, 1, b'ab'), (104, 2, 2, b'ef')])

    def test_conflicting_retransmission_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'conflict'):
            reassemble_tcp([(100, 1, 1, b'abc'), (101, 2, 2, b'XX')])

    def test_deframe_uses_length_not_tcp_packet_boundaries(self):
        frames = list(split_gateway_frames(b'\x02\0\0\x03abc\x01\0\0\x02de'))
        self.assertEqual(frames, [(0, 2, b'abc'), (7, 1, b'de')])

    def test_truncated_or_unknown_frame_is_rejected(self):
        for data in (b'\x02\0\0\x03ab', b'\x02\0', b'\x05\0\0\0'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                list(split_gateway_frames(data))

    def test_tint_id_is_not_brush_id(self):
        msg = {'plots': [{'no': 4, 'facadeLevel': 3, 'items': [
            {'uniqueId': 'paint-a', 'itemId': 201103, 'rawLevel': 3,
             'task': {'effect': {'effectId': 1102, 'selectableEffectId': 3013,
                                 'selectableFromOptIds': [3013, 3010]}}}]}]}
        record = extract_brush_records(msg)[0]
        self.assertEqual(record['brush_effect_id'], 3013)
        self.assertEqual(record['tint_effect_id'], 1102)
        self.assertEqual(record['plot_no'], 4)
        self.assertEqual(record['raw_level'], 3)
        self.assertEqual(record['facade_level'], 3)
        self.assertNotIn('final_level', record)

    def test_missing_brush_is_not_inferred_from_tint(self):
        msg = {'changeItems': [{'uniqueId': 'b', 'task': {'effect': {'effectId': 1011}}}]}
        self.assertEqual(extract_brush_records(msg)[0]['brush_effect_id'], 0)

    def test_exact_and_ambiguous_candidates_are_distinct(self):
        self.assertEqual(compare_brush_ids(1011, [1011], paired=True), 'exact')
        self.assertEqual(compare_brush_ids(2001, [2001, 4006], paired=True), 'candidate_contains')
        self.assertEqual(compare_brush_ids(3013, [3005], paired=True), 'mismatch')

    def test_matching_ids_do_not_prove_event_pairing(self):
        self.assertEqual(compare_brush_ids(1011, [1011], paired=False), 'unpaired')

    def test_missing_values_are_not_counted_as_matches(self):
        self.assertEqual(compare_brush_ids(None, [], paired=True), 'protocol_unavailable')
        self.assertEqual(compare_brush_ids(0, [], paired=True), 'protocol_no_brush')
        self.assertEqual(compare_brush_ids(1011, [], paired=True), 'ocr_unknown')


if __name__ == '__main__':
    unittest.main()
