"""
V15 Enhanced Trading Simulator
Extends V13 simulator with V15 features: ATR-based stops, trailing stops, risk profiles.
"""

import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Handle pandas and numpy imports with error handling
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from ..core.data_fetcher import fetch_prices
from ..core.portable_paths import get_path, get_history_path
from ..risk.volatility import calculate_atr
from ..risk.stop_loss import calculate_stop_loss_distance, calculate_stop_loss_price
from ..risk.trailing_stop import create_trailing_stop, TrailingStop
from ..risk.profiles import RiskProfile, get_profile_config
from ..risk.position_sizing import calculate_position_size_with_profile
from ..risk.exposure_tracker import ExposureTracker, Position as ExposurePosition
from ..risk.equity_monitor import get_equity_monitor
from ..learning.trade_tracker import get_trade_tracker, TradeOutcome
from ..learning.prediction_monitor import get_prediction_monitor
from ..learning.failure_tracker import get_failure_tracker
from ..logging.trade_logger import get_trade_logger
from ..sentiment.override import get_sentiment_override


class Position:
    """Represents an open CFD position with V15 features."""
    
    def __init__(
        self,
        position_id: str,
        ticker: str,
        direction: str,
        entry_price: float,
        quantity: float,
        entry_time: datetime,
        stop_price: float,
        target_price: Optional[float],
        trailing_stop: Optional[TrailingStop],
        confidence: float,
        timeframe: str,
        risk_profile: RiskProfile
    ):
        """Initialize position."""
        self.position_id = position_id
        self.ticker = ticker
        self.direction = direction.upper()
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.stop_price = stop_price
        self.target_price = target_price
        self.trailing_stop = trailing_stop
        self.confidence = confidence
        self.timeframe = timeframe
        self.risk_profile = risk_profile
    
    def update_trailing_stop(self, current_price: float, df: pd.DataFrame) -> bool:
        """Update trailing stop if active."""
        if self.trailing_stop:
            try:
                atr = calculate_atr(df, period=14)
                updated = self.trailing_stop.update(current_price, new_atr=atr)
                if updated:
                    self.stop_price = self.trailing_stop.get_current_stop()
                return updated
            except Exception:
                return False
        return False
    
    def check_exit(self, current_price: float) -> Tuple[bool, str]:
        """
        Check if position should be closed.
        
        Returns:
            Tuple of (should_close, reason)
        """
        # Check trailing stop
        if self.trailing_stop and self.trailing_stop.is_triggered(current_price):
            return (True, "Trailing Stop")
        
        # Check stop-loss
        if self.direction == "LONG":
            if current_price <= self.stop_price:
                return (True, "Stop Loss")
        else:  # SHORT
            if current_price >= self.stop_price:
                return (True, "Stop Loss")
        
        # Check take-profit
        if self.target_price:
            if self.direction == "LONG":
                if current_price >= self.target_price:
                    return (True, "Take Profit")
            else:  # SHORT
                if current_price <= self.target_price:
                    return (True, "Take Profit")
        
        return (False, "")


class TradingSimulatorV15:
    """Enhanced trading simulator with V15 features."""
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_profile: RiskProfile = RiskProfile.MEDIUM
    ):
        """Initialize simulator."""
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_profile = risk_profile
        self.positions: Dict[str, Position] = {}
        self.completed_trades: List[Dict] = []
        
        # V15 components
        self.equity_monitor = get_equity_monitor()
        self.equity_monitor.update_equity(initial_balance)
        self.exposure_tracker = ExposureTracker(initial_balance, risk_profile)
        self.trade_tracker = get_trade_tracker()
        self.trade_logger = get_trade_logger()
        self.prediction_monitor = get_prediction_monitor()
        self.failure_tracker = get_failure_tracker()
        self.sentiment_override = get_sentiment_override()
    
    async def open_position(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        confidence: float,
        timeframe: str,
        df: pd.DataFrame
    ) -> Optional[str]:
        """
        Open a new position with V15 risk management.
        
        Returns:
            Position ID if opened, None otherwise
        """
        # Check sentiment override
        should_block, reason = self.sentiment_override.should_block_trade(ticker)
        if should_block:
            return None
        
        # Calculate stop-loss using ATR
        try:
            stop_distance, atr = calculate_stop_loss_distance(
                df=df,
                profile=self.risk_profile,
                confidence=confidence,
                asset_risk_category="medium"  # Could be enhanced to detect
            )
            stop_price = calculate_stop_loss_price(entry_price, direction, stop_distance)
        except Exception:
            return None
        
        # Calculate position size
        equity = self.equity_monitor.get_current_equity()
        position_size, risk_amount, reason = calculate_position_size_with_profile(
            equity=equity,
            entry_price=entry_price,
            stop_price=stop_price,
            profile=self.risk_profile,
            confidence=confidence,
            direction=direction
        )
        
        if position_size is None:
            return None
        
        # Check exposure limits
        can_open, exposure_reason = self.exposure_tracker.can_open_new_position(risk_amount)
        if not can_open:
            return None
        
        # Create position
        position_id = f"pos_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create trailing stop
        trailing_stop = create_trailing_stop(
            entry_price=entry_price,
            direction=direction,
            initial_stop=stop_price,
            df=df
        )
        
        position = Position(
            position_id=position_id,
            ticker=ticker,
            direction=direction,
            entry_price=entry_price,
            quantity=position_size,
            entry_time=datetime.now(),
            stop_price=stop_price,
            target_price=None,  # Could calculate from prediction
            trailing_stop=trailing_stop,
            confidence=confidence,
            timeframe=timeframe,
            risk_profile=self.risk_profile
        )
        
        self.positions[position_id] = position
        
        # Add to exposure tracker
        exp_position = ExposurePosition(
            position_id=position_id,
            ticker=ticker,
            direction=direction,
            entry_price=entry_price,
            quantity=position_size,
            stop_price=stop_price,
            current_price=entry_price
        )
        self.exposure_tracker.add_position(exp_position)
        
        # Log trade entry
        trade_id = self.trade_logger.log_trade_entry(
            ticker=ticker,
            side=direction,
            size=position_size,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=None,
            confidence=confidence,
            timeframe=timeframe,
            notes=f"Risk profile: {self.risk_profile.value}"
        )
        
        return position_id
    
    async def update_positions(self, price_data: Dict[str, float]) -> List[Dict]:
        """
        Update all positions and check for exits.
        
        Args:
            price_data: Dictionary mapping ticker to current price
            
        Returns:
            List of closed positions
        """
        closed_positions = []
        
        for position_id, position in list(self.positions.items()):
            ticker = position.ticker
            if ticker not in price_data:
                continue
            
            current_price = price_data[ticker]
            
            # Update trailing stop
            # TODO: Would need to fetch df for each ticker
            # For now, just check regular stops
            
            # Check if should exit
            should_close, reason = position.check_exit(current_price)
            
            if should_close:
                # Close position
                closed_pos = await self._close_position(position_id, current_price, reason)
                if closed_pos:
                    closed_positions.append(closed_pos)
        
        return closed_positions
    
    async def _close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str
    ) -> Optional[Dict]:
        """Close a position."""
        if position_id not in self.positions:
            return None
        
        position = self.positions.pop(position_id)
        
        # Calculate P/L
        if position.direction == "LONG":
            pnl = (exit_price - position.entry_price) * position.quantity
        else:  # SHORT
            pnl = (position.entry_price - exit_price) * position.quantity
        
        pnl_percentage = (pnl / (position.entry_price * position.quantity)) * 100.0
        
        # Update balance
        self.balance += pnl
        self.equity_monitor.update_equity(self.balance)
        
        # Remove from exposure tracker
        self.exposure_tracker.remove_position(position_id)
        
        # Check for failure
        if pnl < 0:
            failure = self.failure_tracker.check_trade_failure(
                trade_id=position_id,
                entry_price=position.entry_price,
                exit_price=exit_price,
                position_size=position.quantity,
                direction=position.direction,
                planned_stop_price=position.stop_price
            )
        
        # Log trade exit
        # Find trade_id from logger
        trades = self.trade_logger.get_trades(ticker=position.ticker)
        trade_id = None
        for trade in trades:
            if not trade.get("exit_time") and trade.get("entry_price") == position.entry_price:
                trade_id = trade.get("trade_id") or position_id
                break
        
        if trade_id:
            self.trade_logger.log_trade_exit(
                trade_id=trade_id,
                close_price=exit_price,
                exit_reason=exit_reason,
                pnl=pnl,
                pnl_percentage=pnl_percentage
            )
        
        # Create trade outcome
        outcome = TradeOutcome(
            trade_id=trade_id or position_id,
            ticker=position.ticker,
            direction=position.direction,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            exit_time=datetime.now(),
            exit_price=exit_price,
            exit_reason=exit_reason,
            position_size=position.quantity,
            stop_price=position.stop_price,
            target_price=position.target_price,
            confidence=position.confidence,
            timeframe=position.timeframe,
            pnl=pnl,
            pnl_percentage=pnl_percentage
        )
        
        self.trade_tracker.add_outcome(outcome)
        
        closed_position = {
            "position_id": position_id,
            "ticker": position.ticker,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "quantity": position.quantity,
            "pnl": pnl,
            "pnl_percentage": pnl_percentage,
            "exit_reason": exit_reason
        }
        
        self.completed_trades.append(closed_position)
        
        return closed_position
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics."""
        total_pnl = sum(t.get("pnl", 0) for t in self.completed_trades)
        wins = [t for t in self.completed_trades if t.get("pnl", 0) > 0]
        losses = [t for t in self.completed_trades if t.get("pnl", 0) < 0]
        
        return {
            "initial_balance": self.initial_balance,
            "current_balance": self.balance,
            "total_pnl": total_pnl,
            "total_trades": len(self.completed_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.completed_trades) if self.completed_trades else 0.0,
            "open_positions": len(self.positions),
            "exposure_pct": self.exposure_tracker.get_total_exposure()
        }
    
    def save_simulation(self) -> None:
        """Save simulation state."""
        try:
            history_dir = get_path('history')
            history_dir.mkdir(parents=True, exist_ok=True)
            
            sim_file = history_dir / 'trading_sim_V15.json'
            
            data = {
                "initial_balance": self.initial_balance,
                "current_balance": self.balance,
                "risk_profile": self.risk_profile.value,
                "positions": {pid: {
                    "ticker": pos.ticker,
                    "direction": pos.direction,
                    "entry_price": pos.entry_price,
                    "quantity": pos.quantity,
                    "entry_time": pos.entry_time.isoformat()
                } for pid, pos in self.positions.items()},
                "completed_trades": self.completed_trades,
                "statistics": self.get_statistics()
            }
            
            with open(sim_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

