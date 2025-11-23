"""
Dashboard Page
Overview of key metrics, recent activity, and quick insights.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from logging.trade_logger import get_trade_logger
from logging.analyzer import calculate_performance_metrics, generate_performance_report
from risk.equity_monitor import get_equity_monitor
from risk.exposure_tracker import get_exposure_tracker


def show_dashboard():
    """Display dashboard page."""
    
    st.title("📊 Dashboard")
    st.markdown("---")
    
    # Get data
    trade_logger = get_trade_logger()
    trades = trade_logger.get_trades()
    metrics = calculate_performance_metrics(trades)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total P/L",
            value=f"${metrics.get('total_pnl', 0):,.2f}",
            delta=f"{metrics.get('avg_pnl', 0):.2f} avg"
        )
    
    with col2:
        win_rate = metrics.get('win_rate', 0) * 100
        st.metric(
            label="Win Rate",
            value=f"{win_rate:.1f}%",
            delta=f"{metrics.get('wins', 0)}/{metrics.get('completed_trades', 0)}"
        )
    
    with col3:
        st.metric(
            label="Total Trades",
            value=metrics.get('total_trades', 0),
            delta=f"{metrics.get('open_trades', 0)} open"
        )
    
    with col4:
        st.metric(
            label="Profit Factor",
            value=f"{metrics.get('profit_factor', 0):.2f}",
            delta=f"Max DD: ${metrics.get('max_drawdown', 0):,.2f}"
        )
    
    st.markdown("---")
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Performance Over Time")
        _plot_performance_chart(trades)
    
    with col2:
        st.subheader("🎯 Trade Distribution")
        _plot_trade_distribution(trades)
    
    st.markdown("---")
    
    # Recent Activity and Risk Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Recent Trades")
        _show_recent_trades(trades)
    
    with col2:
        st.subheader("⚠️ Risk Metrics")
        _show_risk_metrics()
    
    st.markdown("---")
    
    # Quick Stats
    st.subheader("📊 Quick Statistics")
    _show_quick_stats(metrics, trades)


def _plot_performance_chart(trades):
    """Plot cumulative P/L over time."""
    if not trades:
        st.info("No trades to display")
        return
    
    completed = [t for t in trades if t.get("exit_time") and t.get("pnl") is not None]
    if not completed:
        st.info("No completed trades to display")
        return
    
    # Sort by exit time
    completed.sort(key=lambda x: x.get("exit_time", ""))
    
    # Calculate cumulative P/L
    cumulative_pnl = []
    running_total = 0.0
    dates = []
    
    for trade in completed:
        running_total += trade.get("pnl", 0)
        cumulative_pnl.append(running_total)
        dates.append(trade.get("exit_time", ""))
    
    # Create DataFrame
    df = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Cumulative P/L": cumulative_pnl
    })
    
    # Plot
    st.line_chart(df.set_index("Date")["Cumulative P/L"])


def _plot_trade_distribution(trades):
    """Plot trade distribution by result."""
    if not trades:
        st.info("No trades to display")
        return
    
    completed = [t for t in trades if t.get("exit_time") and t.get("pnl") is not None]
    if not completed:
        st.info("No completed trades to display")
        return
    
    wins = len([t for t in completed if t.get("pnl", 0) > 0])
    losses = len([t for t in completed if t.get("pnl", 0) < 0])
    breakeven = len([t for t in completed if t.get("pnl", 0) == 0])
    
    data = {
        "Wins": wins,
        "Losses": losses,
        "Breakeven": breakeven
    }
    
    st.bar_chart(data)


def _show_recent_trades(trades, limit=10):
    """Display recent trades table."""
    if not trades:
        st.info("No trades found")
        return
    
    # Sort by entry time (most recent first)
    sorted_trades = sorted(trades, key=lambda x: x.get("entry_time", ""), reverse=True)
    recent = sorted_trades[:limit]
    
    # Prepare data for display
    display_data = []
    for trade in recent:
        entry_time = trade.get("entry_time", "")
        if entry_time:
            try:
                dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                entry_time = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
        
        pnl = trade.get("pnl")
        pnl_display = f"${pnl:,.2f}" if pnl is not None else "Open"
        result = trade.get("result", "Open")
        
        display_data.append({
            "Time": entry_time,
            "Ticker": trade.get("ticker", ""),
            "Side": trade.get("side", ""),
            "Entry": f"${trade.get('entry_price', 0):.2f}",
            "P/L": pnl_display,
            "Result": result
        })
    
    if display_data:
        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No trades to display")


def _show_risk_metrics():
    """Display current risk metrics."""
    try:
        equity_monitor = get_equity_monitor()
        exposure_tracker = get_exposure_tracker()
        
        equity = equity_monitor.get_current_equity()
        exposure = exposure_tracker.get_total_exposure()
        exposure_pct = (exposure / equity * 100) if equity > 0 else 0
        
        st.metric("Current Equity", f"${equity:,.2f}")
        st.metric("Total Exposure", f"${exposure:,.2f} ({exposure_pct:.1f}%)")
        
        # Exposure limit
        max_exposure = equity * 0.10  # 10% max
        st.progress(min(exposure_pct / 10, 1.0))
        st.caption(f"Max Exposure Limit: ${max_exposure:,.2f} (10%)")
        
    except Exception as e:
        st.warning(f"Could not load risk metrics: {e}")


def _show_quick_stats(metrics, trades):
    """Display quick statistics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write("**Completed Trades**")
        st.write(f"{metrics.get('completed_trades', 0)}")
    
    with col2:
        st.write("**Total Profit**")
        st.write(f"${metrics.get('total_profit', 0):,.2f}")
    
    with col3:
        st.write("**Total Loss**")
        st.write(f"${metrics.get('total_loss', 0):,.2f}")
    
    with col4:
        st.write("**Avg Confidence**")
        if trades:
            completed = [t for t in trades if t.get("exit_time")]
            if completed:
                avg_conf = sum(t.get("confidence", 0) for t in completed) / len(completed)
                st.write(f"{avg_conf:.1%}")
            else:
                st.write("N/A")
        else:
            st.write("N/A")


