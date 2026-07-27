"""Stub classifier module — BaseClassifier is only used as an optional type hint
in the bundled cleandiffuser diffusion code.  The demo plugin never passes a
classifier, so a minimal sentinel class is sufficient."""


class BaseClassifier:
    """Placeholder so that ``from cleandiffuser.classifier import BaseClassifier``
    resolves without pulling in the full cleandiffuser package."""
    pass
