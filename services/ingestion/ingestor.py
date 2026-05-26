"""
services/ingestion/ingestor.py

Reads a contract file and returns normalised plain text.
Implemented in Issue 6 alongside agent/clause_discovery.py.
"""

from pathlib import Path


class Ingestor:
    """Reads .txt or .pdf contracts and returns plain text."""

    def ingest(self, path: str | Path) -> str:
        """
        Return the full plain-text content of the contract at path.
        Raises FileNotFoundError if the path does not exist.
        Raises ValueError for unsupported file types.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()
        if suffix == ".txt":
            return path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".pdf":
            return self._read_pdf(path)
        raise ValueError(f"Unsupported file type: {suffix}. Expected .txt or .pdf")

    def _read_pdf(self, path: Path) -> str:
        # Implemented in Issue 6
        raise NotImplementedError("PDF ingestion not yet implemented")
