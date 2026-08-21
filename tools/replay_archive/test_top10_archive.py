import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.replay_archive.top10_archive import (
    atomic_write_json,
    extract_team_names_from_prefix,
    pending_source_days,
    rank_daily_teams,
    read_replay_bytes_from_zip,
    scan_team_names_from_zip,
    select_episode_ids_by_team_names,
    source_row_digest,
    validate_archive,
    write_archive_manifest,
)


class ReplayArchiveCoreTests(unittest.TestCase):
    def test_extract_team_names_from_small_replay_prefix(self):
        prefix = b'{"id":1,"info":{"EpisodeId":1,"TeamNames":["Alpha","Beta"],"x":1},"steps":['
        self.assertEqual(extract_team_names_from_prefix(prefix), ("Alpha", "Beta"))

    def test_extract_team_names_handles_json_metacharacters_inside_names(self):
        names = ["Alpha ] bracket", 'Beta "quote" \ slash カ']
        prefix = (
            '{"id":1,"info":{"EpisodeId":1,"TeamNames":'
            + json.dumps(names, ensure_ascii=False)
            + ',"x":1},"steps":['
        ).encode("utf-8")
        self.assertEqual(extract_team_names_from_prefix(prefix), tuple(names))

    def test_extract_team_names_rejects_missing_header(self):
        with self.assertRaisesRegex(RuntimeError, "TeamNames"):
            extract_team_names_from_prefix(b'{"info":{"EpisodeId":1}}')

    def test_scan_zip_skips_missing_and_malformed_replay_members(self):
        rows = [
            {"episode_id": "1", "avg_score": "1300", "min_score": "1250"},
            {"episode_id": "2", "avg_score": "1200", "min_score": "1180"},
            {"episode_id": "3", "avg_score": "1100", "min_score": "1080"},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "day.zip"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("1.json", '{"info":{"TeamNames":["A","B"]},"steps":[]}')
                zf.writestr("2.json", '{"info":{"EpisodeId":2},"steps":[]}')
                # 3.json is intentionally absent although episode 3 is in the manifest.
            with zipfile.ZipFile(path) as zf:
                entries, names, skipped = scan_team_names_from_zip(zf, rows, header_scan_bytes=4096)

        self.assertEqual(set(entries), {1, 2})
        self.assertEqual(names, {1: ("A", "B")})
        self.assertEqual(
            skipped,
            [
                {"episode_id": 2, "reason": "invalid_header"},
                {"episode_id": 3, "reason": "missing_member"},
            ],
        )

    def test_daily_rank_skips_unavailable_and_invalid_score_rows(self):
        rows = [
            {"episode_id": "1", "avg_score": "1300", "min_score": "1250"},
            {"episode_id": "2", "avg_score": "1200", "min_score": "1180"},
            {"episode_id": "3", "avg_score": "not-a-number", "min_score": "1000"},
        ]
        names = {1: ("A", "B"), 3: ("C", "D")}
        ranked = rank_daily_teams(rows, names, n=10)
        self.assertEqual([r["team_name"] for r in ranked], ["A", "B"])

    def test_read_replay_bytes_skips_missing_and_non_utf8_members(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "day.zip"
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("1.json", b'{"info":{"TeamNames":["A","B"]},"steps":[]}')
                zf.writestr("2.json", b'\xff\xfe\xfd')
            with zipfile.ZipFile(path) as zf:
                entries = {int(Path(i.filename).stem): i for i in zf.infolist()}
                raw, reason = read_replay_bytes_from_zip(zf, entries, 1)
                self.assertIsNone(reason)
                self.assertIn(b'"TeamNames"', raw)
                raw, reason = read_replay_bytes_from_zip(zf, entries, 2)
                self.assertIsNone(raw)
                self.assertEqual(reason, "invalid_utf8")
                raw, reason = read_replay_bytes_from_zip(zf, entries, 3)
                self.assertIsNone(raw)
                self.assertEqual(reason, "missing_member")

    def test_daily_rank_uses_team_score_evidence_not_manifest_order(self):
        rows = [
            {"episode_id": "3", "avg_score": "1000", "min_score": "900"},
            {"episode_id": "1", "avg_score": "1300", "min_score": "1250"},
            {"episode_id": "2", "avg_score": "1200", "min_score": "1180"},
        ]
        names = {
            1: ("A", "B"),
            2: ("C", "D"),
            3: ("E", "F"),
        }
        ranked = rank_daily_teams(rows, names, n=4)
        self.assertEqual([r["team_name"] for r in ranked], ["A", "B", "C", "D"])
        self.assertEqual(ranked[0]["score_proxy"], 1300.0)
        self.assertIn("max observed avg_score", ranked[0]["ranking_basis"])

    def test_daily_rank_uses_best_observation_for_team(self):
        rows = [
            {"episode_id": "1", "avg_score": "1000", "min_score": "900"},
            {"episode_id": "2", "avg_score": "1400", "min_score": "1350"},
            {"episode_id": "3", "avg_score": "1300", "min_score": "1290"},
        ]
        names = {1: ("A", "B"), 2: ("A", "C"), 3: ("D", "E")}
        ranked = rank_daily_teams(rows, names, n=5)
        a = next(r for r in ranked if r["team_name"] == "A")
        self.assertEqual(a["score_proxy"], 1400.0)
        self.assertEqual(a["evidence_episode_id"], 2)
        self.assertEqual(a["appearances"], 2)

    def test_selection_is_unique_and_sorted(self):
        names = {7: ("A", "B"), 5: ("C", "A"), 8: ("D", "E")}
        self.assertEqual(select_episode_ids_by_team_names(names, {"A"}), [5, 7])

    def test_pending_source_days_is_idempotent_and_rejects_mutation(self):
        rows = [
            {"date": "2026-08-01", "daily_dataset_slug": "d1", "episode_count": "10"},
            {"date": "2026-08-02", "daily_dataset_slug": "d2", "episode_count": "12"},
        ]
        processed = {"2026-08-01": {"source_row_sha256": source_row_digest(rows[0]), "status": "complete"}}
        self.assertEqual([r["date"] for r in pending_source_days(rows, processed)], ["2026-08-02"])
        changed = [dict(rows[0], episode_count="11"), rows[1]]
        with self.assertRaisesRegex(RuntimeError, "changed after processing"):
            pending_source_days(changed, processed)

    def test_archive_manifest_detects_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atomic_write_json(root / "processed_days.json", {})
            manifest = write_archive_manifest(
                root,
                processed_dates=[],
                episode_count=0,
                top10_row_count=0,
                source_index_manifest_sha256="abc",
            )
            self.assertEqual(validate_archive(root)["manifest_sha256"], manifest["manifest_sha256"])
            (root / "processed_days.json").write_text('{"tampered":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                validate_archive(root)

    def test_dataset_metadata_is_not_corpus_hashed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atomic_write_json(root / "processed_days.json", {})
            (root / "dataset-metadata.json").write_text('{"title":"x"}\n', encoding="utf-8")
            write_archive_manifest(
                root,
                processed_dates=[],
                episode_count=0,
                top10_row_count=0,
                source_index_manifest_sha256="abc",
            )
            (root / "dataset-metadata.json").write_text('{"title":"y"}\n', encoding="utf-8")
            validate_archive(root)

    def test_atomic_json_is_canonical(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.json"
            atomic_write_json(path, {"b": 2, "a": 1})
            first = path.read_bytes()
            atomic_write_json(path, {"a": 1, "b": 2})
            self.assertEqual(first, path.read_bytes())
            self.assertEqual(json.loads(first), {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
