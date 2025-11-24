"""
Trade History Page
Trade history viewing, filtering, and analysis with performance metrics.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sa_logging.trade_logger import get_trade_logger
from sa_logging.analyzer import calculate_performance_metrics, generate_performance_report
from core.timeframes import ALL_TIMEFRAMES


def show_trade_history():
    """Display trade history page."""
    
    st.title("📜 Trade History")
    st.markdown("---")
    
    # Get all trades
    try:
        trade_logger = get_trade_logger()
        all_trades = trade_logger.get_trades()
        
        if not all_trades:
            st.info("No trades found. Trades will appear here once logged.")
            return
        
        # Filters section
        filtered_trades = _apply_filters(all_trades)
        
        # Performance metrics
        _display_performance_metrics(filtered_trades)
        
        st.markdown("---")
        
        # Charts
        _display_charts(filtered_trades)
        
        st.markdown("---")
        
        # Trade table
        _display_trade_table(filtered_trades)
        
        st.markdown("---")
        
        # Export functionality
        _export_trades(filtered_trades)
        
    except Exception as e:
        st.error(f"❌ Error loading trade history: {str(e)}")
        st.exception(e)


def _apply_filters(all_trades: List[Dict]) -> List[Dict]:
    """Apply filters to trades."""
    st.subheader("🔍 Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Ticker filter
        all_tickers = sorted(set(t.get("ticker", "") for t in all_trades if t.get("ticker")))
        selected_ticker = st.selectbox(
            "Ticker",
            options=["All"] + all_tickers,
            index=0
        )
    
    with col2:
        # Result filter
        result_options = ["All", "Win", "Loss", "Breakeven", "Open"]
        selected_result = st.selectbox("Result", options=result_options, index=0)
    
    with col3:
        # Timeframe filter
        timeframe_options = ["All"] + ALL_TIMEFRAMES
        selected_timeframe = st.selectbox("Timeframe", options=timeframe_options, index=0)
    
    with col4:
        # Date range filter
        date_filter = st.selectbox(
            "Date Range",
            options=["All", "Last 7 days", "Last 30 days", "Last 90 days", "Last year"],
            index=0
        )
    
    # Apply filters
    filtered = all_trades.copy()
    
    # Ticker filter
    if selected_ticker != "All":
        filtered = [t for t in filtered if t.get("ticker", "") == selected_ticker]
    
    # Result filter
    if selected_result != "All":
        if selected_result == "Open":
            filtered = [t for t in filtered if not t.get("exit_time")]
        else:
            filtered = [t for t in filtered if t.get("result", "") == selected_result]
    
    # Timeframe filter
    if selected_timeframe != "All":
        filtered = [t for t in filtered if t.get("timeframe", "") == selected_timeframe]
    
    # Date range filter
    if date_filter != "All":
        now = datetime.now()
        if date_filter == "Last 7 days":
            cutoff = now - timedelta(days=7)
        elif date_filter == "Last 30 days":
            cutoff = now - timedelta(days=30)
        elif date_filter == "Last 90 days":
            cutoff = now - timedelta(days=90)
        elif date_filter == "Last year":
            cutoff = now - timedelta(days=365)
        else:
            cutoff = None
        
        if cutoff:
            filtered = [
                t for t in filtered
                if t.get("entry_time") and _parse_datetime(t.get("entry_time")) >= cutoff
            ]
    
    st.caption(f"Showing {len(filtered)} of {len(all_trades)} trades")
    
    return filtered


def _parse_datetime(dt_str: str) -> Optional[datetime]:
    """Parse datetime string."""
    if not dt_str:
        return None
    try:
        # Handle ISO format with or without timezone
        dt_str = dt_str.replace('Z', '+00:00')
        return datetime.fromisoformat(dt_str)
    except:
        try:
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except:
            return None


def _display_performance_metrics(trades: List[Dict]):
    """Display performance metrics."""
    st.subheader("📊 Performance Metrics")
    
    if not trades:
        st.info("No trades to analyze")
        return
    
    metrics = calculate_performance_metrics(trades)
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Trades", metrics.get('total_trades', 0))
        st.caption(f"{metrics.get('open_trades', 0)} open")
    
    with col2:
        win_rate = metrics.get('win_rate', 0) * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")
        st.caption(f"{metrics.get('wins', 0)} wins")
    
    with col3:
        st.metric("Total P/L", f"${metrics.get('total_pnl', 0):,.2f}")
        st.caption(f"Avg: ${metrics.get('avg_pnl', 0):,.2f}")
    
    with col4:
        profit_factor = metrics.get('profit_factor', 0)
        st.metric("Profit Factor", f"{profit_factor:.2f}")
        st.caption(f"Profit: ${metrics.get('total_profit', 0):,.2f}")
    
    with col5:
        max_dd = metrics.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"${max_dd:,.2f}")
        st.caption(f"{metrics.get('max_drawdown_pct', 0):.2f}%")
    
    # Additional stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Completed Trades**")
        st.write(f"{metrics.get('completed_trades', 0)}")
    
    with col2:
        st.write("**Total Profit**")
        st.write(f"${metrics.get('total_profit', 0):,.2f}")
    
    with col3:
        st.write("**Total Loss**")
        st.write(f"${metrics.get('total_loss', 0):,.2f}")


def _display_charts(trades: List[Dict]):
    """Display performance charts."""
    st.subheader("📈 Performance Charts")
    
    if not trades:
        st.info("No trades to display")
        return
    
    # Filter completed trades
    completed = [t for t in trades if t.get("exit_time") and t.get("pnl") is not None]
    
    if not completed:
        st.info("No completed trades to display")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Cumulative P/L Over Time**")
        _plot_cumulative_pnl(completed)
    
    with col2:
        st.write("**P/L Distribution**")
        _plot_pnl_distribution(completed)
    
    # Win/Loss breakdown
    st.write("**Win/Loss Breakdown**")
    _plot_win_loss_breakdown(completed)


def _plot_cumulative_pnl(trades: List[Dict]):
    """Plot cumulative P/L over time."""
    if not trades:
        return
    
    # Sort by exit time
    sorted_trades = sorted(trades, key=lambda x: x.get("exit_time", ""))
    
    # Calculate cumulative P/L
    cumulative_pnl = []
    running_total = 0.0
    dates = []
    
    for trade in sorted_trades:
        pnl = trade.get("pnl", 0)
        running_total += pnl
        cumulative_pnl.append(running_total)
        exit_time = trade.get("exit_time", "")
        dates.append(_parse_datetime(exit_time) or datetime.now())
    
    # Create DataFrame
    df = pd.DataFrame({
        "Date": dates,
        "Cumulative P/L": cumulative_pnl
    })
    
    df = df.sort_values("Date")
    st.line_chart(df.set_index("Date")["Cumulative P/L"])


def _plot_pnl_distribution(trades: List[Dict]):
    """Plot P/L distribution."""
    if not trades:
        return
    
    pnl_values = [t.get("pnl", 0) for t in trades]
    
    # Create histogram data
    df = pd.DataFrame({"P/L": pnl_values})
    st.bar_chart(df["P/L"])


def _plot_win_loss_breakdown(trades: List[Dict]):
    """Plot win/loss breakdown."""
    if not trades:
        return
    
    wins = len([t for t in trades if t.get("pnl", 0) > 0])
    losses = len([t for t in trades if t.get("pnl", 0) < 0])
    breakeven = len([t for t in trades if t.get("pnl", 0) == 0])
    
    data = {
        "Wins": wins,
        "Losses": losses,
        "Breakeven": breakeven
    }
    
    st.bar_chart(data)


def _display_trade_table(trades: List[Dict]):
    """Display trade table."""
    st.subheader("📋 Trade Details")
    
    if not trades:
        st.info("No trades to display")
        return
    
    # Prepare table data
    table_data = []
    for trade in trades:
        entry_time = trade.get("entry_time", "")
        exit_time = trade.get("exit_time", "")
        
        # Format dates
        entry_dt = _parse_datetime(entry_time)
        exit_dt = _parse_datetime(exit_time)
        
        entry_str = entry_dt.strftime("%Y-%m-%d %H:%M") if entry_dt else "N/A"
        exit_str = exit_dt.strftime("%Y-%m-%d %H:%M") if exit_dt else "Open"
        
        pnl = trade.get("pnl")
        pnl_display = f"${pnl:,.2f}" if pnl is not None else "Open"
        
        result = trade.get("result", "Open")
        confidence = trade.get("confidence", 0)
        
        table_data.append({
            "Entry Time": entry_str,
            "Exit Time": exit_str,
            "Ticker": trade.get("ticker", ""),
            "Side": trade.get("side", ""),
            "Entry Price": f"${trade.get('entry_price', 0):.2f}",
            "Exit Price": f"${trade.get('close_price', 0):.2f}" if trade.get("close_price") else "Open",
            "Size": f"{trade.get('size', 0):.2f}",
            "P/L": pnl_display,
            "P/L %": f"{trade.get('pnl_percentage', 0):.2f}%" if trade.get("pnl_percentage") is not None else "N/A",
            "Result": result,
            "Confidence": f"{confidence:.1%}" if confidence > 0 else "N/A",
            "Timeframe": trade.get("timeframe", ""),
            "Exit Reason": trade.get("exit_reason", "")
        })
    
    df = pd.DataFrame(table_data)
    
    # Sort by entry time (most recent first)
    if "Entry Time" in df.columns:
        df = df.sort_values("Entry Time", ascending=False)
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Trade statistics by ticker
    if len(trades) > 0:
        st.write("**Statistics by Ticker**")
        _display_ticker_statistics(trades)


def _display_ticker_statistics(trades: List[Dict]):
    """Display statistics grouped by ticker."""
    ticker_stats = {}
    
    for trade in trades:
        ticker = trade.get("ticker", "Unknown")
        if ticker not in ticker_stats:
            ticker_stats[ticker] = {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0
            }
        
        ticker_stats[ticker]["total"] += 1
        pnl = trade.get("pnl", 0)
        ticker_stats[ticker]["total_pnl"] += pnl
        
        if pnl > 0:
            ticker_stats[ticker]["wins"] += 1
        elif pnl < 0:
            ticker_stats[ticker]["losses"] += 1
    
    # Create DataFrame
    stats_data = []
    for ticker, stats in ticker_stats.items():
        win_rate = (stats["wins"] / stats["total"] * 100) if stats["total"] > 0 else 0
        stats_data.append({
            "Ticker": ticker,
            "Total Trades": stats["total"],
            "Wins": stats["wins"],
            "Losses": stats["losses"],
            "Win Rate": f"{win_rate:.1f}%",
            "Total P/L": f"${stats['total_pnl']:,.2f}"
        })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)


def _export_trades(trades: List[Dict]):
    """Export trades functionality."""
    st.subheader("💾 Export Trades")
    
    if not trades:
        st.info("No trades to export")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        if st.button("Export to CSV", use_container_width=True):
            _export_to_csv(trades)
    
    with col2:
        # JSON export
        if st.button("Export to JSON", use_container_width=True):
            _export_to_json(trades)


def _export_to_csv(trades: List[Dict]):
    """Export trades to CSV."""
    try:
        # Prepare CSV data
        csv_data = []
        for trade in trades:
            csv_data.append({
                "Entry Time": trade.get("entry_time", ""),
                "Exit Time": trade.get("exit_time", ""),
                "Ticker": trade.get("ticker", ""),
                "Side": trade.get("side", ""),
                "Entry Price": trade.get("entry_price", 0),
                "Exit Price": trade.get("close_price", 0),
                "Size": trade.get("size", 0),
                "Stop Price": trade.get("stop_price", 0),
                "P/L": trade.get("pnl", 0),
                "P/L %": trade.get("pnl_percentage", 0),
                "Result": trade.get("result", ""),
                "Confidence": trade.get("confidence", 0),
                "Timeframe": trade.get("timeframe", ""),
                "Exit Reason": trade.get("exit_reason", ""),
                "Notes": trade.get("notes", "")
            })
        
        df = pd.DataFrame(csv_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"❌ Error exporting to CSV: {str(e)}")


def _export_to_json(trades: List[Dict]):
    """Export trades to JSON."""
    try:
        import json
        
        json_data = json.dumps(trades, indent=2, default=str)
        
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    except Exception as e:
        st.error(f"❌ Error exporting to JSON: {str(e)}")

