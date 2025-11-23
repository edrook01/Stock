"""
Cache Management System
Monitors cache size, prunes old files, and maintains cache health.
"""

import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

from .portable_paths import get_path

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages cache files and disk space."""
    
    def __init__(self):
        """Initialize cache manager."""
        self.cache_dir = get_path('cache')
        self.memory_dir = get_path('memory')
        self.cache_db_path = self.memory_dir / 'cache.db'
    
    def get_cache_size(self) -> Dict[str, any]:
        """
        Get cache size statistics.
        
        Returns:
            Dictionary with cache statistics:
            {
                "total_size_mb": float,
                "file_count": int,
                "oldest_file": str,
                "newest_file": str,
                "cache_db_size_mb": float,
                "breakdown": Dict[str, float]  # Size by directory
            }
        """
        total_size = 0.0
        file_count = 0
        oldest_file = None
        newest_file = None
        oldest_time = None
        newest_time = None
        breakdown = {}
        
        # Check cache directory
        if self.cache_dir.exists():
            cache_size, cache_count, cache_oldest, cache_newest = self._get_directory_stats(self.cache_dir)
            total_size += cache_size
            file_count += cache_count
            breakdown["cache"] = cache_size
            
            if cache_oldest and (oldest_time is None or cache_oldest < oldest_time):
                oldest_time = cache_oldest
                oldest_file = str(self.cache_dir / cache_oldest)
            if cache_newest and (newest_time is None or cache_newest > newest_time):
                newest_time = cache_newest
                newest_file = str(self.cache_dir / cache_newest)
        
        # Check memory directory (for cache.db and other files)
        if self.memory_dir.exists():
            memory_size, memory_count, memory_oldest, memory_newest = self._get_directory_stats(self.memory_dir)
            total_size += memory_size
            file_count += memory_count
            breakdown["memory"] = memory_size
            
            if memory_oldest and (oldest_time is None or memory_oldest < oldest_time):
                oldest_time = memory_oldest
                oldest_file = str(self.memory_dir / memory_oldest)
            if memory_newest and (newest_time is None or memory_newest > newest_time):
                newest_time = memory_newest
                newest_file = str(self.memory_dir / memory_newest)
        
        # Check cache.db specifically
        cache_db_size = 0.0
        if self.cache_db_path.exists():
            cache_db_size = self.cache_db_path.stat().st_size / (1024 * 1024)
            breakdown["cache_db"] = cache_db_size
        
        return {
            "total_size_mb": total_size,
            "file_count": file_count,
            "oldest_file": oldest_file or "N/A",
            "newest_file": newest_file or "N/A",
            "cache_db_size_mb": cache_db_size,
            "breakdown": breakdown
        }
    
    def _get_directory_stats(
        self,
        directory: Path
    ) -> Tuple[float, int, Optional[datetime], Optional[datetime]]:
        """
        Get statistics for a directory.
        
        Returns:
            Tuple of (total_size_mb, file_count, oldest_file_time, newest_file_time)
        """
        total_size = 0.0
        file_count = 0
        oldest_time = None
        newest_time = None
        
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    try:
                        stat = file_path.stat()
                        size_mb = stat.st_size / (1024 * 1024)
                        total_size += size_mb
                        file_count += 1
                        
                        mtime = datetime.fromtimestamp(stat.st_mtime)
                        if oldest_time is None or mtime < oldest_time:
                            oldest_time = mtime
                        if newest_time is None or mtime > newest_time:
                            newest_time = mtime
                    except (OSError, PermissionError):
                        continue
        except Exception as e:
            logger.warning(f"Error scanning directory {directory}: {e}")
        
        return (total_size, file_count, oldest_time, newest_time)
    
    def prune_cache(
        self,
        max_age_days: int = 30,
        max_size_mb: float = 1000.0,
        dry_run: bool = False
    ) -> Dict[str, any]:
        """
        Prune cache files based on age and size.
        
        Args:
            max_age_days: Maximum age in days for cached files
            max_size_mb: Maximum total cache size in MB
            dry_run: If True, don't actually delete files
            
        Returns:
            Dictionary with pruning results:
            {
                "removed_count": int,
                "freed_mb": float,
                "remaining_size_mb": float,
                "files_removed": List[str]
            }
        """
        removed_count = 0
        freed_mb = 0.0
        files_removed = []
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Get current cache size
        stats = self.get_cache_size()
        current_size = stats["total_size_mb"]
        
        # First, remove files older than max_age_days
        for directory in [self.cache_dir, self.memory_dir]:
            if not directory.exists():
                continue
            
            for file_path in directory.rglob('*'):
                if file_path.is_file() and file_path.name != 'cache.db':  # Don't delete cache.db
                    try:
                        stat = file_path.stat()
                        file_time = datetime.fromtimestamp(stat.st_mtime)
                        file_size_mb = stat.st_size / (1024 * 1024)
                        
                        if file_time < cutoff_date:
                            if not dry_run:
                                file_path.unlink()
                            removed_count += 1
                            freed_mb += file_size_mb
                            files_removed.append(str(file_path))
                    except (OSError, PermissionError):
                        continue
        
        # If still over max_size_mb, remove oldest files (FIFO)
        remaining_size = current_size - freed_mb
        if remaining_size > max_size_mb:
            # Collect all files with their modification times
            files_with_times = []
            for directory in [self.cache_dir, self.memory_dir]:
                if not directory.exists():
                    continue
                
                for file_path in directory.rglob('*'):
                    if file_path.is_file() and file_path.name != 'cache.db':
                        try:
                            stat = file_path.stat()
                            files_with_times.append((
                                file_path,
                                datetime.fromtimestamp(stat.st_mtime),
                                stat.st_size / (1024 * 1024)
                            ))
                        except (OSError, PermissionError):
                            continue
            
            # Sort by modification time (oldest first)
            files_with_times.sort(key=lambda x: x[1])
            
            # Remove oldest files until under limit
            for file_path, file_time, file_size in files_with_times:
                if remaining_size <= max_size_mb:
                    break
                
                if not dry_run:
                    try:
                        file_path.unlink()
                    except (OSError, PermissionError):
                        continue
                
                removed_count += 1
                freed_mb += file_size
                remaining_size -= file_size
                files_removed.append(str(file_path))
        
        return {
            "removed_count": removed_count,
            "freed_mb": round(freed_mb, 2),
            "remaining_size_mb": round(remaining_size, 2),
            "files_removed": files_removed[:50]  # Limit to first 50 for reporting
        }
    
    def clear_cache(
        self,
        confirm: bool = False
    ) -> bool:
        """
        Clear all cache files.
        
        Args:
            confirm: Must be True to actually clear
            
        Returns:
            True if cache was cleared, False otherwise
        """
        if not confirm:
            return False
        
        removed_count = 0
        
        # Clear cache directory
        if self.cache_dir.exists():
            for file_path in self.cache_dir.rglob('*'):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except (OSError, PermissionError):
                        continue
        
        # Clear memory directory cache files (but not cache.db or other important files)
        if self.memory_dir.exists():
            important_files = {'cache.db', 'ticker_validation_cache.json'}
            for file_path in self.memory_dir.iterdir():
                if file_path.is_file() and file_path.name not in important_files:
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except (OSError, PermissionError):
                        continue
        
        logger.info(f"Cleared {removed_count} cache files")
        return True
    
    def get_cache_statistics(self) -> Dict[str, any]:
        """
        Get detailed cache statistics.
        
        Returns:
            Dictionary with statistics:
            {
                "total_size_mb": float,
                "file_count": int,
                "average_file_age_days": float,
                "oldest_file_age_days": float,
                "newest_file_age_days": float,
                "size_by_type": Dict[str, float],
                "recommendations": List[str]
            }
        """
        stats = self.get_cache_size()
        
        # Calculate file ages
        now = datetime.now()
        oldest_file_path = stats.get("oldest_file")
        newest_file_path = stats.get("newest_file")
        
        oldest_age_days = 0.0
        newest_age_days = 0.0
        
        if oldest_file_path and oldest_file_path != "N/A":
            try:
                oldest_stat = Path(oldest_file_path).stat()
                oldest_age_days = (now - datetime.fromtimestamp(oldest_stat.st_mtime)).total_seconds() / 86400
            except:
                pass
        
        if newest_file_path and newest_file_path != "N/A":
            try:
                newest_stat = Path(newest_file_path).stat()
                newest_age_days = (now - datetime.fromtimestamp(newest_stat.st_mtime)).total_seconds() / 86400
            except:
                pass
        
        # Generate recommendations
        recommendations = []
        if stats["total_size_mb"] > 2000:
            recommendations.append("Cache size exceeds 2GB. Consider pruning old files.")
        if oldest_age_days > 90:
            recommendations.append(f"Oldest cache file is {oldest_age_days:.0f} days old. Consider pruning.")
        if stats["file_count"] > 10000:
            recommendations.append(f"Cache has {stats['file_count']} files. Consider pruning.")
        
        return {
            "total_size_mb": round(stats["total_size_mb"], 2),
            "file_count": stats["file_count"],
            "average_file_age_days": round((oldest_age_days + newest_age_days) / 2, 1) if oldest_age_days > 0 else 0,
            "oldest_file_age_days": round(oldest_age_days, 1),
            "newest_file_age_days": round(newest_age_days, 1),
            "size_by_type": stats.get("breakdown", {}),
            "recommendations": recommendations
        }


# Global cache manager instance
_cache_manager_instance: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = CacheManager()
    return _cache_manager_instance

