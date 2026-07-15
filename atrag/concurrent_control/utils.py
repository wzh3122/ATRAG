"""
Utility functions for the concurrent control system.

This module contains helper functions and context managers for working
with locks in a more convenient way.
"""

from contextlib import asynccontextmanager
from typing import AsyncContextManager, Optional

from .protocols import LockProtocol


class LockAcquisitionError(Exception):
    """Raised when a lock cannot be acquired."""

    pass


@asynccontextmanager
async def lock_context(lock: LockProtocol, timeout: Optional[float] = None) -> AsyncContextManager[None]:
    """
    Convenient async context manager for operations that need locking.

    Usage:
        async with lock_context(my_lock):
            # Your protected operations here
            await some_critical_operation()

    Args:
        lock: Lock instance to use
        timeout: Maximum time to wait for lock acquisition
    """
    success = await lock.acquire(timeout=timeout)
    if not success:
        lock_name = lock.get_name()
        raise TimeoutError(f"Failed to acquire lock '{lock_name}' within {timeout} seconds")

    try:
        yield
    finally:
        await lock.release()
