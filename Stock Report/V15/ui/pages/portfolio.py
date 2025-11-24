"""
Portfolio Page
Portfolio management and position tracking with exposure monitoring.
"""

import streamlit as st
import pandas as pd
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sa_logging.trade_logger import get_trade_logger
from risk.equity_monitor import get_equity_monitor
from risk.exposure_tracker import ExposureTracker, Position
from risk.profiles import get_risk_profile, RiskProfile, get_max_combined_exposure
from risk.position_sizing import calculate_position_size_with_profile
from core.data_fetcher import fetch_prices


def show_portfolio():
    """Display portfolio page."""
    
    st.title("💼 Portfolio")
    st.markdown("---")
    
    # Get current equity and exposure tracker
    try:
        equity_monitor = get_equity_monitor()
        current_equity = equity_monitor.get_current_equity()
        
        # Initialize exposure tracker if needed
        # Get risk profile from config
        from core.portable_paths import get_data_path
        import json
        try:
            config_file = get_data_path() / 'config_v15.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                profile_name = config.get("risk_profile", "medium")
                risk_profile = get_risk_profile(profile_name) or RiskProfile.MEDIUM
            else:
                risk_profile = RiskProfile.MEDIUM
        except:
            risk_profile = RiskProfile.MEDIUM
        
        # Get or create exposure tracker in session state
        if 'exposure_tracker' not in st.session_state:
            st.session_state.exposure_tracker = ExposureTracker(current_equity, risk_profile)
        
        exposure_tracker = st.session_state.exposure_tracker
        exposure_tracker.update_equity(current_equity)
        
        # Load positions from trade logs
        _load_positions_from_logs(exposure_tracker)
        
        # Display equity and exposure overview
        _display_equity_overview(equity_monitor, exposure_tracker)
        
        st.markdown("---")
        
        # Positions table
        _display_positions_table(exposure_tracker, current_equity)
        
        st.markdown("---")
        
        # Position management
        with st.expander("➕ Add Position Manually"):
            _add_position_manual(exposure_tracker, current_equity)
        
        # Update prices
        if exposure_tracker.positions:
            with st.expander("🔄 Update Position Prices"):
                _update_position_prices(exposure_tracker)
        
    except Exception as e:
        st.error(f"❌ Error loading portfolio: {str(e)}")
        st.exception(e)


def _show_equity_setup(equity_monitor):
    """Show equity setup form."""
    st.subheader("Account Setup")
    
    current_equity = equity_monitor.get_current_equity()
    
    with st.form("equity_setup"):
        new_equity = st.number_input(
            "Account Equity ($)",
            min_value=0.0,
            value=current_equity if current_equity > 0 else 10000.0,
            step=100.0
        )
        
        submitted = st.form_submit_button("Set Equity", type="primary")
        
        if submitted:
            equity_monitor.update_equity(new_equity)
            st.success(f"✅ Equity set to ${new_equity:,.2f}")
            st.rerun()


def _display_equity_overview(equity_monitor, exposure_tracker: ExposureTracker):
    """Display equity and exposure overview."""
    st.subheader("📊 Portfolio Overview")
    
    current_equity = equity_monitor.get_current_equity()
    total_exposure_pct = exposure_tracker.get_total_exposure()
    total_exposure_amount = exposure_tracker.get_worst_case_loss()
    max_exposure_pct = get_max_combined_exposure(exposure_tracker.profile)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Equity", f"${current_equity:,.2f}")
    
    with col2:
        st.metric("Total Exposure", f"{total_exposure_pct:.2f}%")
        st.caption(f"${total_exposure_amount:,.2f}")
    
    with col3:
        max_exposure_amount = current_equity * (max_exposure_pct / 100.0)
        remaining = max_exposure_amount - total_exposure_amount
        st.metric("Exposure Limit", f"{max_exposure_pct:.1f}%")
        st.caption(f"Remaining: ${remaining:,.2f}")
    
    with col4:
        risk_profile = exposure_tracker.profile
        st.metric("Risk Profile", risk_profile.value.upper())
    
    # Exposure progress bar
    st.markdown("**Exposure Usage**")
    exposure_ratio = total_exposure_pct / max_exposure_pct if max_exposure_pct > 0 else 0
    st.progress(min(exposure_ratio, 1.0))
    
    # Warning if approaching limit
    if total_exposure_pct >= max_exposure_pct * 0.8:
        st.warning(f"⚠️ Exposure approaching limit ({total_exposure_pct:.2f}% / {max_exposure_pct:.1f}%)")
    elif total_exposure_pct >= max_exposure_pct:
        st.error(f"🚫 Exposure limit exceeded ({total_exposure_pct:.2f}% / {max_exposure_pct:.1f}%)")
    
    # Drawdown info
    drawdown, drawdown_pct = equity_monitor.get_drawdown()
    max_dd, max_dd_pct = equity_monitor.get_max_drawdown()
    
    if drawdown > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Drawdown", f"${drawdown:,.2f}", f"{drawdown_pct:.2f}%")
        with col2:
            st.metric("Max Drawdown", f"${max_dd:,.2f}", f"{max_dd_pct:.2f}%")


def _load_positions_from_logs(exposure_tracker: ExposureTracker):
    """Load open positions from trade logs."""
    try:
        trade_logger = get_trade_logger()
        trades = trade_logger.get_trades()
        
        # Filter open trades (no exit_time)
        open_trades = [t for t in trades if not t.get("exit_time")]
        
        # Add positions to tracker
        for trade in open_trades:
            position_id = trade.get("trade_id")
            if not position_id or position_id in exposure_tracker.positions:
                continue
            
            ticker = trade.get("ticker", "")
            side = trade.get("side", "LONG")
            entry_price = trade.get("entry_price", 0.0)
            stop_price = trade.get("stop_price", 0.0)
            size = trade.get("size", 0.0)
            current_price = entry_price  # Default to entry price if not updated
            
            if entry_price > 0 and stop_price > 0 and size > 0:
                position = Position(
                    position_id=position_id,
                    ticker=ticker,
                    direction=side,
                    entry_price=entry_price,
                    quantity=size,
                    stop_price=stop_price,
                    current_price=current_price
                )
                exposure_tracker.add_position(position)
    
    except Exception as e:
        st.warning(f"⚠️ Could not load positions from logs: {str(e)}")


def _display_positions_table(exposure_tracker: ExposureTracker, equity: float):
    """Display positions table."""
    st.subheader("📋 Open Positions")
    
    positions = list(exposure_tracker.positions.values())
    
    if not positions:
        st.info("No open positions. Add positions manually or they will appear here when trades are logged.")
        return
    
    # Prepare table data
    table_data = []
    for pos in positions:
        # Calculate P/L
        if pos.direction == "LONG":
            pnl = (pos.current_price - pos.entry_price) * pos.quantity
            pnl_pct = ((pos.current_price - pos.entry_price) / pos.entry_price * 100) if pos.entry_price > 0 else 0
        else:  # SHORT
            pnl = (pos.entry_price - pos.current_price) * pos.quantity
            pnl_pct = ((pos.entry_price - pos.current_price) / pos.entry_price * 100) if pos.entry_price > 0 else 0
        
        # Calculate exposure
        risk_amount = pos.get_risk_amount()
        exposure_pct = pos.get_exposure_percentage(equity)
        
        table_data.append({
            "Position ID": pos.position_id[:8] + "...",  # Truncate for display
            "Ticker": pos.ticker,
            "Side": pos.direction,
            "Entry Price": f"${pos.entry_price:.2f}",
            "Current Price": f"${pos.current_price:.2f}",
            "Quantity": f"{pos.quantity:.2f}",
            "Stop-Loss": f"${pos.stop_price:.2f}",
            "P/L": f"${pnl:,.2f}",
            "P/L %": f"{pnl_pct:+.2f}%",
            "Risk": f"${risk_amount:,.2f}",
            "Exposure %": f"{exposure_pct:.2f}%"
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Summary stats
    total_pnl = sum(
        (pos.current_price - pos.entry_price) * pos.quantity if pos.direction == "LONG"
        else (pos.entry_price - pos.current_price) * pos.quantity
        for pos in positions
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Positions", len(positions))
    with col2:
        st.metric("Total Unrealized P/L", f"${total_pnl:,.2f}")
    with col3:
        total_risk = exposure_tracker.get_worst_case_loss()
        st.metric("Total Risk", f"${total_risk:,.2f}")


def _add_position_manual(exposure_tracker: ExposureTracker, equity: float):
    """Form to add position manually."""
    with st.form("add_position"):
        col1, col2 = st.columns(2)
        
        with col1:
            ticker = st.text_input("Ticker", value="", placeholder="AAPL").upper().strip()
            direction = st.selectbox("Direction", ["LONG", "SHORT"])
            entry_price = st.number_input("Entry Price ($)", min_value=0.01, value=100.0, step=0.01)
            current_price = st.number_input("Current Price ($)", min_value=0.01, value=100.0, step=0.01)
        
        with col2:
            quantity = st.number_input("Quantity", min_value=0.01, value=1.0, step=0.01)
            stop_price = st.number_input("Stop-Loss Price ($)", min_value=0.01, value=95.0, step=0.01)
        
        submitted = st.form_submit_button("Add Position", type="primary")
        
        if submitted:
            if not ticker:
                st.error("❌ Please enter a ticker symbol")
                return
            
            if entry_price <= 0 or stop_price <= 0 or quantity <= 0:
                st.error("❌ Prices and quantity must be positive")
                return
            
            # Validate stop price
            if direction == "LONG" and stop_price >= entry_price:
                st.error("❌ Stop-loss must be below entry price for LONG positions")
                return
            if direction == "SHORT" and stop_price <= entry_price:
                st.error("❌ Stop-loss must be above entry price for SHORT positions")
                return
            
            # Check exposure limit
            position = Position(
                position_id=f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                ticker=ticker,
                direction=direction,
                entry_price=entry_price,
                quantity=quantity,
                stop_price=stop_price,
                current_price=current_price
            )
            
            new_risk = position.get_risk_amount()
            can_open, reason = exposure_tracker.can_open_new_position(new_risk)
            
            if not can_open:
                st.error(f"❌ Cannot add position: {reason}")
                return
            
            # Add position
            exposure_tracker.add_position(position)
            st.success(f"✅ Position added: {ticker} {direction}")
            st.rerun()


def _update_position_prices(exposure_tracker: ExposureTracker):
    """Update current prices for positions."""
    positions = list(exposure_tracker.positions.values())
    
    if not positions:
        st.info("No positions to update")
        return
    
    # Create update form
    with st.form("update_prices"):
        st.write("**Update Current Prices**")
        
        price_updates = {}
        for pos in positions:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.text(f"{pos.ticker} {pos.direction} (Entry: ${pos.entry_price:.2f})")
            with col2:
                new_price = st.number_input(
                    "Current Price",
                    min_value=0.01,
                    value=pos.current_price,
                    step=0.01,
                    key=f"price_{pos.position_id}"
                )
                price_updates[pos.position_id] = new_price
        
        submitted = st.form_submit_button("Update Prices", type="primary")
        
        if submitted:
            updated_count = 0
            for position_id, new_price in price_updates.items():
                if exposure_tracker.update_position_price(position_id, new_price):
                    updated_count += 1
            
            if updated_count > 0:
                st.success(f"✅ Updated {updated_count} position(s)")
                st.rerun()
            else:
                st.warning("⚠️ No positions updated")



