import json
import tempfile
import unittest
from pathlib import Path

from tools.replay_archive.top10_archive import (
    apply_episode_updates,
    atomic_write_json,
    daily_top_teams,
    merge_episode_metadata,
    merge_episode_rows,
    normalize_episode_payload,
    pending_source_days,
    select_episode_ids,
    source_row_digest,
    validate_archive,
    write_archive_manifest,
)


class ReplayArchiveCoreTests(unittest.TestCase):
    def _episode(self, episode_id=101):
        return {
            "id": episode_id,
            "createTime": "2026-08-01T01:00:00Z",
            "endTime": "2026-08-01T01:03:00Z",
            "agents": [
                {"teamId": 7, "teamName": "Alpha", "submissionId": 7001, "initialScore": 1500.0, "updatedScore": 1512.5, "reward": 10},
                {"teamId": 8, "teamName": "Beta", "submissionId": 8001, "initialScore": 1500.0, "updatedScore": 1487.5, "reward": 5},
            ],
        }

    def test_normalize_episode_payload_preserves_skill_ratings(self):
        rows = normalize_episode_payload({"episodes": [self._episode()]})
        self.assertEqual(rows[0]["episode_id"], 101)
        self.assertEqual(rows[0]["agents"][0]["updated_score"], 1512.5)
        self.assertEqual(rows[0]["agents"][0]["team_id"], 7)

    def test_normalize_episode_payload_accepts_nested_result(self):
        rows = normalize_episode_payload({"result": {"episodes": [self._episode(102)]}})
        self.assertEqual(rows[0]["episode_id"], 102)

    def test_merge_episode_metadata_deduplicates_team_histories(self):
        a = normalize_episode_payload({"episodes": [self._episode(201)]})
        b = normalize_episode_payload({"episodes": [self._episode(201), self._episode(202)]})
        merged = merge_episode_metadata(a, b)
        self.assertEqual([x["episode_id"] for x in merged], [201, 202])

    def test_merge_episode_metadata_rejects_conflict(self):
        a = normalize_episode_payload({"episodes": [self._episode(201)]})
        changed = self._episode(201)
        changed["agents"][0]["updatedScore"] = 9999
        b = normalize_episode_payload({"episodes": [changed]})
        with self.assertRaisesRegex(RuntimeError, "conflicting episode metadata"):
            merge_episode_metadata(a, b)

    def test_daily_rank_uses_best_of_two_newest_submissions(self):
        state = {}
        episodes = normalize_episode_payload({"episodes": [
            {"id": 1, "endTime": "2026-08-01T01:00:00Z", "agents": [
                {"teamId": 1, "teamName": "A", "submissionId": 11, "updatedScore": 2100.0},
                {"teamId": 2, "teamName": "B", "submissionId": 21, "updatedScore": 2050.0},
            ]},
            {"id": 2, "endTime": "2026-08-02T01:00:00Z", "agents": [
                {"teamId": 1, "teamName": "A", "submissionId": 12, "updatedScore": 1900.0},
                {"teamId": 3, "teamName": "C", "submissionId": 31, "updatedScore": 2000.0},
            ]},
            {"id": 3, "endTime": "2026-08-03T01:00:00Z", "agents": [
                {"teamId": 1, "teamName": "A", "submissionId": 13, "updatedScore": 1800.0},
                {"teamId": 4, "teamName": "D", "submissionId": 41, "updatedScore": 1700.0},
            ]},
        ]})
        apply_episode_updates(state, episodes)
        ranked = daily_top_teams(state, n=4)
        by_team = {r["team_id"]: r for r in ranked}
        self.assertEqual(by_team[1]["score"], 1900.0)
        self.assertEqual(by_team[1]["submission_id"], 12)
        self.assertEqual([r["team_id"] for r in ranked], [2, 3, 1, 4])

    def test_select_episode_ids_is_unique_and_sorted(self):
        episodes = [
            {"episode_id": 7, "agents": [{"team_id": 1}, {"team_id": 2}]},
            {"episode_id": 5, "agents": [{"team_id": 9}, {"team_id": 1}]},
            {"episode_id": 7, "agents": [{"team_id": 1}, {"team_id": 2}]},
            {"episode_id": 8, "agents": [{"team_id": 9}, {"team_id": 10}]},
        ]
        self.assertEqual(select_episode_ids(episodes, {1, 2}), [5, 7])

    def test_pending_source_days_skips_exact_rows_and_rejects_mutation(self):
        rows = [
            {"date": "2026-08-01", "daily_dataset_slug": "d1", "episode_count": "10"},
            {"date": "2026-08-02", "daily_dataset_slug": "d2", "episode_count": "12"},
        ]
        processed = {"2026-08-01": {"source_row_sha256": source_row_digest(rows[0]), "status": "complete"}}
        self.assertEqual([r["date"] for r in pending_source_days(rows, processed)], ["2026-08-02"])
        changed = [dict(rows[0], episode_count="11"), rows[1]]
        with self.assertRaisesRegex(RuntimeError, "changed after processing"):
            pending_source_days(changed, processed)

    def test_episode_manifest_merge_is_idempotent(self):
        existing = [
            {"episode_id": 1, "date": "2026-08-01", "replay_sha256": "aaa"},
            {"episode_id": 2, "date": "2026-08-01", "replay_sha256": "bbb"},
        ]
        incoming = [
            {"episode_id": 2, "date": "2026-08-01", "replay_sha256": "bbb"},
            {"episode_id": 3, "date": "2026-08-02", "replay_sha256": "ccc"},
        ]
        once = merge_episode_rows(existing, incoming)
        twice = merge_episode_rows(once, incoming)
        self.assertEqual([r["episode_id"] for r in once], [1, 2, 3])
        self.assertEqual(once, twice)

    def test_episode_manifest_merge_rejects_conflicting_duplicate(self):
        existing = [{"episode_id": 1, "date": "2026-08-01", "replay_sha256": "aaa"}]
        incoming = [{"episode_id": 1, "date": "2026-08-01", "replay_sha256": "DIFFERENT"}]
        with self.assertRaisesRegex(RuntimeError, "conflicting episode metadata"):
            merge_episode_rows(existing, incoming)

    def test_archive_manifest_detects_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "state").mkdir()
            atomic_write_json(root / "state/rating_state.json", {"1": {"team_id": 1}})
            atomic_write_json(root / "state/processed_days.json", {})
            manifest = write_archive_manifest(root, processed_dates=[], episode_count=0, top10_row_count=0)
            self.assertEqual(validate_archive(root)["manifest_sha256"], manifest["manifest_sha256"])
            (root / "state/rating_state.json").write_text('{"tampered":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                validate_archive(root)

    def test_atomic_json_is_canonical_and_reproducible(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.json"
            atomic_write_json(path, {"b": 2, "a": 1})
            first = path.read_bytes()
            atomic_write_json(path, {"a": 1, "b": 2})
            self.assertEqual(first, path.read_bytes())
            self.assertEqual(json.loads(first), {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
