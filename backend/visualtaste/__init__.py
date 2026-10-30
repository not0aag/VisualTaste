"""VisualTaste: a local, single-user image recommendation engine.

The package is organised by concern:

    config      one settings object, resolved once, used everywhere
    db          SQLite schema, migrations and the query layer
    data        acquisition, validation, derivatives, dedupe, safety, selection
    vision      frozen image encoders and the embedding store
    search      FAISS indexes over those embeddings
    rec         the recommendation ladder R0-R6 and the feed that serves it
    eval        the simulator, the metrics and the replay harness
    viz         2-D projections for the embedding map
    api         the FastAPI application
    jobs        command-line entry points for everything long-running
"""

__version__ = "0.1.0"
