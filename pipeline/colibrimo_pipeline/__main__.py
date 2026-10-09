"""Allow ``python -m colibrimo_pipeline ...`` as an alternative to ``cpipe ...``."""

from colibrimo_pipeline.cli import main

raise SystemExit(main())
