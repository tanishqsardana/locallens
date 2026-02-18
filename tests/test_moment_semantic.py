from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from videosearch.video_cycle import (
    build_moment_semantic_embeddings,
    persist_moment_index,
    semantic_search_moments,
)


class MomentSemanticTest(unittest.TestCase):
    def test_build_and_search_semantic_index(self) -> None:
        moments = [
            {
                "moment_index": 0,
                "video_id": "v1",
                "type": "APPEAR",
                "start_time": 1.0,
                "end_time": 1.0,
                "entities": [10],
                "metadata": {"label": "truck", "label_group": "truck"},
            },
            {
                "moment_index": 1,
                "video_id": "v1",
                "type": "APPEAR",
                "start_time": 2.0,
                "end_time": 2.0,
                "entities": [20],
                "metadata": {"label": "car", "label_group": "car"},
            },
        ]
        keyframes = []
        image_embeddings = {
            0: np.ones(8, dtype=np.float32),
            1: np.ones(8, dtype=np.float32),
        }
        semantic_texts, semantic_embeddings, semantic_model = build_moment_semantic_embeddings(
            moments,
            embedder_kind="hashing",
            embedder_model=None,
        )
        self.assertEqual(len(semantic_texts), 2)
        self.assertEqual(len(semantic_embeddings), 2)
        self.assertTrue(semantic_model.startswith("hashing-"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "moment_index.sqlite"
            persist_moment_index(
                output_db=db_path,
                moments=moments,
                keyframes=keyframes,
                embeddings=image_embeddings,
                model_name="hist",
                semantic_texts=semantic_texts,
                semantic_embeddings=semantic_embeddings,
                semantic_model_name=semantic_model,
            )
            results = semantic_search_moments(db_path, query="when does truck appear", top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["metadata"].get("label"), "truck")


if __name__ == "__main__":
    unittest.main()
