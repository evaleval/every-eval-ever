"""HELM-specific core helpers (namespace).

This package contains modules that are specific to downloading and scraping
HELM evaluation data. Keeping them under `src.core.helm` makes it clearer
when other sources are added in the future.
"""

__all__ = ["web_scraper", "downloader", "scrape_all_to_parquet"]
