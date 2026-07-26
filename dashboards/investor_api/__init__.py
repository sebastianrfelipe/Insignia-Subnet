"""Read-only investor reporting API (SPEC §8). Publication is gated by
otc.compliance; internal generation is not."""

from dashboards.investor_api.factsheet import Factsheet, build_factsheet, render_markdown

__all__ = ["Factsheet", "build_factsheet", "render_markdown"]
