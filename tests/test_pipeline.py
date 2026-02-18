from __future__ import annotations

import tempfile
import unittest

from videosearch.chunking import ChunkConfig
from videosearch.embedders import HashingEmbedder
from videosearch.pipeline import DomainChunkingPolicy, VideoSemanticSearch
from videosearch.store import SqliteVectorStore
from videosearch.types import TranscriptSegment, VideoTranscript


class VideoSemanticSearchTest(unittest.TestCase):
    def test_index_and_search(self) -> None:
        transcripts = [
            VideoTranscript(
                video_id="fin-1",
                title="Risk Controls",
                domain="finance",
                segments=[
                    TranscriptSegment(0, 10, "Diversification reduces concentration risk in portfolios."),
                    TranscriptSegment(10, 20, "Rebalancing positions keeps exposure aligned with target allocation."),
                    TranscriptSegment(20, 30, "Risk tolerance should drive leverage decisions."),
                ],
            ),
            VideoTranscript(
                video_id="med-1",
                title="Diabetes Care",
                domain="medical",
                segments=[
                    TranscriptSegment(0, 10, "Glucose monitoring guides medication and nutrition changes."),
                    TranscriptSegment(10, 20, "Exercise improves insulin sensitivity over time."),
                    TranscriptSegment(20, 30, "A1C trends support long-term treatment planning."),
                ],
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/index.sqlite"
            store = SqliteVectorStore(db_path)
            store.initialize(recreate=True)
            app = VideoSemanticSearch(store=store, embedder=HashingEmbedder())
            count = app.index_transcripts(
                transcripts=transcripts,
                policy=DomainChunkingPolicy(
                    default=ChunkConfig(max_words=40, stride_words=20, min_words=1),
                    per_domain={},
                ),
                batch_size=8,
            )
            self.assertGreaterEqual(count, 2)

            results = app.search("How should I rebalance my portfolio risk?", top_k=3, domain="finance")
            self.assertTrue(results)
            self.assertEqual(results[0].chunk.domain, "finance")


if __name__ == "__main__":
    unittest.main()
