"""Audio similarity search over PostgreSQL + pgvector.

The package is deliberately split so that the two retrieval strategies compared in
the report stay independent of each other:

``features.embedding``  dense MFCC statistics, searched with pgvector (Path A)
``features.landmarks``  Shazam-style spectral peak pairs, searched with a B-tree (Path B)

Everything else (database access, ingestion, querying, evaluation) is shared.
"""

__version__ = "1.0.0"
