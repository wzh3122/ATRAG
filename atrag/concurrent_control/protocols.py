"""
Protocol definitions for the concurrent control system.

This module contains the abstract interfaces that all lock implementations
must follow to ensure compatibility across different deployment scenarios.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class LockProtocol(ABC):
    """
    Abstract interface for concurrent locks.

    This protocol defines the common interface that all lock implementations
    must follow to ensure compatibility across different deployment scenarios.
    """

    @abstractmethod
    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the lock asynchronously.

        Args:
            timeout: Maximum time to wait for the lock (seconds).
                    None means wait indefinitely.

        Returns:
            True if lock was acquired successfully, False if timeout occurred.
        """
        pass

    @abstractmethod
    async def release(self) -> None:
        """Release the lock asynchronously."""
        pass

    @abstractmethod
    def is_locked(self) -> bool:
        """Check if the lock is currently held."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name/identifier of the lock."""
        pass

    @abstractmethod
    async def __aenter__(self) -> "LockProtocol":
        """Async context manager entry."""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        pass
