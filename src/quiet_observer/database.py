from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from .config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from . import models  # noqa: F401 - import models to register them
    Base.metadata.create_all(bind=engine)

    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text(
                "ALTER TABLE projects ADD COLUMN last_inferred_frame_id INTEGER"
            ))
            conn.commit()
        except Exception:
            conn.rollback()

        try:
            conn.execute(text(
                "ALTER TABLE frames ADD COLUMN source VARCHAR DEFAULT 'sampler'"
            ))
            conn.commit()
        except Exception:
            conn.rollback()

        for old, new in [
            ("capture_interval_seconds", "sample_interval_seconds"),
            ("capture_active", "sampling_active"),
            ("last_capture_at", "last_sample_at"),
        ]:
            try:
                conn.execute(text(
                    f"ALTER TABLE projects RENAME COLUMN {old} TO {new}"
                ))
                conn.commit()
            except Exception:
                conn.rollback()

        try:
            conn.execute(text(
                "UPDATE frames SET source = 'sampler' WHERE source = 'capture'"
            ))
            conn.commit()
        except Exception:
            conn.rollback()

        # Migrate is_labeled boolean to label_status string
        try:
            conn.execute(text(
                "ALTER TABLE frames ADD COLUMN label_status VARCHAR DEFAULT 'unlabeled'"
            ))
            conn.commit()
        except Exception:
            conn.rollback()

        try:
            conn.execute(text(
                "UPDATE frames SET label_status = 'annotated' WHERE is_labeled = 1 AND label_status = 'unlabeled'"
            ))
            conn.commit()
        except Exception:
            conn.rollback()
