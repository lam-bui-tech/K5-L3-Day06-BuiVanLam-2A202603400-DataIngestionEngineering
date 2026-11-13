"""Store and search vectors using PostgreSQL + pgvector."""
import json
from collections import defaultdict
from typing import Any
import psycopg
from psycopg.rows import dict_row

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        """Retain connection settings; validate model dimensions."""
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        """Create table with vector column and source/page/filter metadata."""
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        strategy VARCHAR(32) NOT NULL,
                        chunk_id VARCHAR(128) NOT NULL,
                        document_id VARCHAR(64) NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}',
                        embedding vector({self.dimensions}) NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        PRIMARY KEY (strategy, chunk_id)
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_strat ON rag_chunks(strategy);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc ON rag_chunks(strategy, document_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_meta ON rag_chunks USING gin(metadata);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_vec ON rag_chunks USING hnsw (embedding vector_cosine_ops);")
            conn.commit()

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]], batch_size: int = 100) -> None:
        """Validate lengths/dimensions and write stable IDs without duplicates.

        When re-ingesting changed source documents, remove obsolete chunks too.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"Length mismatch: {len(chunks)} chunks vs {len(vectors)} vectors")
        if not chunks:
            return

        for i, vec in enumerate(vectors):
            if len(vec) != self.dimensions:
                raise ValueError(
                    f"Vector dimension mismatch at index {i}: expected {self.dimensions}, got {len(vec)}"
                )

        # Group chunks by document_id to clean up obsolete chunks per document
        docs_to_chunks = defaultdict(list)
        for chunk, vec in zip(chunks, vectors):
            docs_to_chunks[chunk.document_id].append((chunk, vec))

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                for doc_id, doc_items in docs_to_chunks.items():
                    current_ids = [c.chunk_id for c, _ in doc_items]
                    # Clean up obsolete chunks for this document in this strategy
                    cur.execute(
                        """
                        DELETE FROM rag_chunks
                        WHERE strategy = %s AND document_id = %s AND NOT (chunk_id = ANY(%s));
                        """,
                        (strategy, doc_id, current_ids),
                    )

                # Upsert all items in batches
                insert_sql = """
                    INSERT INTO rag_chunks (
                        strategy, chunk_id, document_id, source, page, text, metadata, embedding
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s::vector
                    ) ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        source = EXCLUDED.source,
                        page = EXCLUDED.page,
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        created_at = NOW();
                """

                rows_to_insert = []
                for chunk, vec in zip(chunks, vectors):
                    vec_str = "[" + ",".join(f"{x:.8f}" for x in vec) + "]"
                    meta_json = json.dumps(chunk.metadata, ensure_ascii=False)
                    rows_to_insert.append((
                        strategy, chunk.chunk_id, chunk.document_id,
                        chunk.source, chunk.page, chunk.text, meta_json, vec_str
                    ))

                for i in range(0, len(rows_to_insert), batch_size):
                    batch = rows_to_insert[i:i + batch_size]
                    cur.executemany(insert_sql, batch)

            conn.commit()

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Filter permitted metadata in SQL, rank by cosine distance, return top-k.

        Return chunk ID, text, score and metadata. Parameterize user-supplied values.
        """
        # Strategy C re-uses Strategy B's index with metadata filter applied
        target_strategy = "B" if strategy == "C" else strategy

        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.dimensions}, got {len(query_vector)}"
            )

        where_clauses = ["strategy = %s"]
        params: list[Any] = [target_strategy]

        if filters:
            for k, v in filters.items():
                if v is not None:
                    where_clauses.append("metadata->>%s = %s")
                    params.extend([str(k), str(v)])

        where_sql = " AND ".join(where_clauses)
        vec_str = "[" + ",".join(f"{x:.8f}" for x in query_vector) + "]"

        query_sql = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   (embedding <=> %s::vector) AS distance
            FROM rag_chunks
            WHERE {where_sql}
            ORDER BY distance ASC
            LIMIT %s;
        """
        full_params = [vec_str] + params + [top_k]

        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(query_sql, full_params)
                rows = cur.fetchall()

        results = []
        for r in rows:
            dist = float(r["distance"])
            sim_score = max(0.0, 1.0 - dist)
            results.append({
                "chunk_id": r["chunk_id"],
                "document_id": r["document_id"],
                "source": r["source"],
                "page": r["page"],
                "text": r["text"],
                "metadata": r["metadata"],
                "distance": dist,
                "score": sim_score,
            })

        return results

    def count_chunks(self, strategy: str) -> int:
        """Count total chunks stored for a strategy."""
        target_strategy = "B" if strategy == "C" else strategy
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM rag_chunks WHERE strategy = %s;", (target_strategy,))
                res = cur.fetchone()
                return int(res[0]) if res else 0
