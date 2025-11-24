"""
Ticker List Auditor
Audits ticker lists, identifies invalid/delisted tickers, and manages ticker mappings.
"""

import asyncio
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .portable_paths import get_path, get_data_path
from .ticker_validator import get_ticker_validator
import logging

# Set up logging
logger = logging.getLogger(__name__)


class TickerAuditor:
    """Audits and manages ticker lists."""
    
    def __init__(self):
        """Initialize ticker auditor."""
        self.validator = get_ticker_validator()
        self.mappings_file = get_data_path() / 'ticker_mappings.json'
        self.delisted_file = get_data_path() / 'delisted_tickers.json'
        self.audit_log_file = get_path('logs') / 'ticker_audit.log'
        
        # Ensure log directory exists
        self.audit_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load mappings and delisted lists
        self.mappings = self._load_mappings()
        self.delisted = self._load_delisted()
    
    async def audit_ticker_list(
        self,
        ticker_list: List[str],
        auto_fix: bool = False
    ) -> Dict[str, Any]:
        """
        Audit a ticker list and identify issues.
        
        Args:
            ticker_list: List of ticker symbols to audit
            auto_fix: Whether to automatically apply fixes
            
        Returns:
            Dictionary with audit results:
            {
                "total": int,
                "valid": int,
                "invalid": int,
                "delisted": int,
                "renamed": int,
                "report": str,
                "valid_tickers": List[str],
                "invalid_tickers": List[str],
                "delisted_tickers": List[str],
                "renamed_tickers": Dict[str, str]
            }
        """
        # Normalize tickers
        normalized_tickers = [t.upper().strip() for t in ticker_list]
        unique_tickers = list(set(normalized_tickers))
        
        # Batch validate all tickers
        validation_results = await self.validator.batch_validate_tickers(unique_tickers)
        
        # Categorize results
        valid_tickers = []
        invalid_tickers = []
        delisted_tickers = []
        renamed_tickers = {}
        
        for ticker, result in validation_results.items():
            if result["valid"]:
                valid_tickers.append(ticker)
            else:
                status = result.get("status", "unknown")
                
                # Check if ticker is in delisted list
                if ticker in self.delisted:
                    delisted_tickers.append(ticker)
                    self._log_ticker_change("DELISTED", ticker, "Previously marked as delisted")
                # Check if ticker has been renamed
                elif ticker in self.mappings:
                    new_ticker = self.mappings[ticker]
                    renamed_tickers[ticker] = new_ticker
                    self._log_ticker_change("RENAMED", ticker, f"Renamed to {new_ticker}", new_ticker)
                else:
                    invalid_tickers.append(ticker)
                    self._log_ticker_change("INVALID", ticker, f"Status: {status}")
        
        # Generate report
        report_lines = [
            f"Ticker Audit Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            f"Total tickers audited: {len(unique_tickers)}",
            f"Valid tickers: {len(valid_tickers)}",
            f"Invalid tickers: {len(invalid_tickers)}",
            f"Delisted tickers: {len(delisted_tickers)}",
            f"Renamed tickers: {len(renamed_tickers)}",
            ""
        ]
        
        if invalid_tickers:
            report_lines.append("Invalid Tickers:")
            for ticker in invalid_tickers[:20]:  # Limit to first 20
                report_lines.append(f"  - {ticker}")
            if len(invalid_tickers) > 20:
                report_lines.append(f"  ... and {len(invalid_tickers) - 20} more")
            report_lines.append("")
        
        if delisted_tickers:
            report_lines.append("Delisted Tickers:")
            for ticker in delisted_tickers[:20]:
                report_lines.append(f"  - {ticker}")
            if len(delisted_tickers) > 20:
                report_lines.append(f"  ... and {len(delisted_tickers) - 20} more")
            report_lines.append("")
        
        if renamed_tickers:
            report_lines.append("Renamed Tickers:")
            for old_ticker, new_ticker in list(renamed_tickers.items())[:20]:
                report_lines.append(f"  - {old_ticker} -> {new_ticker}")
            if len(renamed_tickers) > 20:
                report_lines.append(f"  ... and {len(renamed_tickers) - 20} more")
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        result = {
            "total": len(unique_tickers),
            "valid": len(valid_tickers),
            "invalid": len(invalid_tickers),
            "delisted": len(delisted_tickers),
            "renamed": len(renamed_tickers),
            "report": report,
            "valid_tickers": valid_tickers,
            "invalid_tickers": invalid_tickers,
            "delisted_tickers": delisted_tickers,
            "renamed_tickers": renamed_tickers
        }
        
        # Auto-fix if requested
        if auto_fix:
            cleaned_tickers = valid_tickers.copy()
            # Add renamed tickers (use new symbol)
            for old_ticker, new_ticker in renamed_tickers.items():
                if new_ticker not in cleaned_tickers:
                    cleaned_tickers.append(new_ticker)
            result["cleaned_tickers"] = cleaned_tickers
        
        return result
    
    def flag_delisted_tickers(
        self,
        tickers: List[str]
    ) -> Dict[str, List[str]]:
        """
        Flag tickers as delisted.
        
        Args:
            tickers: List of ticker symbols to flag as delisted
            
        Returns:
            Dictionary with categorized tickers
        """
        normalized = [t.upper().strip() for t in tickers]
        
        # Add to delisted list
        for ticker in normalized:
            if ticker not in self.delisted:
                self.delisted.append(ticker)
                self._log_ticker_change("DELISTED", ticker, "Manually flagged as delisted")
        
        self._save_delisted()
        
        return {
            "delisted": self.delisted.copy(),
            "renamed": {k: v for k, v in self.mappings.items() if k in normalized},
            "valid": [t for t in normalized if t not in self.delisted and t not in self.mappings]
        }
    
    def add_ticker_mapping(
        self,
        old_ticker: str,
        new_ticker: str,
        reason: str = "renamed"
    ) -> None:
        """
        Add a ticker mapping (for renamed/merged tickers).
        
        Args:
            old_ticker: Old ticker symbol
            new_ticker: New ticker symbol
            reason: Reason for mapping (renamed, merged, split)
        """
        old_ticker = old_ticker.upper().strip()
        new_ticker = new_ticker.upper().strip()
        
        self.mappings[old_ticker] = new_ticker
        self._save_mappings()
        self._log_ticker_change("RENAMED", old_ticker, f"{reason} -> {new_ticker}", new_ticker)
    
    async def update_ticker_list(
        self,
        file_path: Path,
        remove_invalid: bool = True,
        output_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Update a ticker list file by removing invalid tickers.
        
        Args:
            file_path: Path to ticker list file
            remove_invalid: Whether to remove invalid tickers
            output_file: Optional output file path (defaults to input file)
            
        Returns:
            Dictionary with update results
        """
        # Load ticker list
        tickers = self._load_ticker_list(file_path)
        
        # Audit the list
        audit_result = await self.audit_ticker_list(tickers, auto_fix=True)
        
        if remove_invalid and "cleaned_tickers" in audit_result:
            # Save cleaned list
            output_path = output_file or file_path
            self._save_ticker_list(output_path, audit_result["cleaned_tickers"])
            
            return {
                "original_count": len(tickers),
                "cleaned_count": len(audit_result["cleaned_tickers"]),
                "removed_count": len(tickers) - len(audit_result["cleaned_tickers"]),
                "audit_result": audit_result
            }
        
        return audit_result
    
    def suggest_alternatives(
        self,
        ticker: str,
        max_suggestions: int = 5
    ) -> List[str]:
        """
        Suggest alternative tickers for a delisted/invalid ticker.
        
        Args:
            ticker: Ticker symbol to find alternatives for
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggested ticker symbols
        """
        ticker = ticker.upper().strip()
        
        suggestions = []
        
        # Check if ticker has a mapping
        if ticker in self.mappings:
            suggestions.append(self.mappings[ticker])
        
        # TODO: Implement more sophisticated suggestions:
        # - Similar ticker names
        # - Same sector/industry
        # - Related companies
        
        return suggestions[:max_suggestions]
    
    def _load_ticker_list(self, file_path: Path) -> List[str]:
        """Load ticker list from file."""
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "tickers" in data:
                        return data["tickers"]
                    elif isinstance(data, list):
                        return data
            else:
                # Assume text file, one ticker per line
                with open(file_path, 'r') as f:
                    return [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error loading ticker list: {e}")
            return []
    
    def _save_ticker_list(
        self,
        file_path: Path,
        tickers: List[str],
        format: str = "json"
    ) -> None:
        """Save ticker list to file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json" or file_path.suffix.lower() == '.json':
            data = {
                "tickers": sorted(tickers),
                "last_updated": datetime.now().isoformat(),
                "total": len(tickers)
            }
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == "csv":
            import csv
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Ticker"])
                for ticker in sorted(tickers):
                    writer.writerow([ticker])
        else:
            # Text file
            with open(file_path, 'w') as f:
                for ticker in sorted(tickers):
                    f.write(f"{ticker}\n")
    
    def _load_mappings(self) -> Dict[str, str]:
        """Load ticker mappings from file."""
        try:
            if self.mappings_file.exists():
                with open(self.mappings_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def _save_mappings(self) -> None:
        """Save ticker mappings to file."""
        try:
            self.mappings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.mappings_file, 'w') as f:
                json.dump(self.mappings, f, indent=2)
        except Exception:
            pass
    
    def _load_delisted(self) -> List[str]:
        """Load delisted tickers from file."""
        try:
            if self.delisted_file.exists():
                with open(self.delisted_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return []
    
    def _save_delisted(self) -> None:
        """Save delisted tickers to file."""
        try:
            self.delisted_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.delisted_file, 'w') as f:
                json.dump(sorted(self.delisted), f, indent=2)
        except Exception:
            pass
    
    def _log_ticker_change(
        self,
        action: str,
        ticker: str,
        reason: str,
        new_ticker: Optional[str] = None
    ) -> None:
        """Log ticker change to audit log."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {action}: {ticker}"
            if new_ticker:
                log_entry += f" -> {new_ticker}"
            log_entry += f" - {reason}\n"
            
            with open(self.audit_log_file, 'a') as f:
                f.write(log_entry)
        except Exception:
            pass


# Global auditor instance
_auditor_instance: Optional[TickerAuditor] = None


def get_ticker_auditor() -> TickerAuditor:
    """Get global ticker auditor instance."""
    global _auditor_instance
    if _auditor_instance is None:
        _auditor_instance = TickerAuditor()
    return _auditor_instance

