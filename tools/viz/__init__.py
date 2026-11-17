"""Exporters for the demo visualizer (card P6-V). They write the data bundles that `site/` reads.

`build_site.py` writes the public bundle to `site/data/public/`. `build_replay.py` writes the local
bundle to `site/data/local/`. The schema is in `site/data/SCHEMA.md`. This package is the only
package that imports both `engine` and `tools.gt`. The public bundle never holds positions or
full-match labels of ours. `guard.py` checks that rule on every public file.
"""
