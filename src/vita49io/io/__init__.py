"""Provide I/O helpers used to create and serialize VITA 49 streams.

Args:
    None.

Returns:
    None.

Raises:
    None.

Side Effects:
    Imports streaming utilities into the subpackage namespace.

Examples:
    >>> from vita49io.io import IQStreamWriter
    >>> isinstance(IQStreamWriter, type)
    True
"""

from .iq_writer import IQStreamWriter

__all__ = ["IQStreamWriter"]

