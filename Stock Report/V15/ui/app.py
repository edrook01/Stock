"""
Stock Analyzer V15 - Web UI
Modern web interface for stock analysis, portfolio management, and trade tracking.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.pages.dashboard import show_dashboard
from ui.pages.stock_analysis import show_stock_analysis
from ui.pages.portfolio import show_portfolio
from ui.pages.trade_history import show_trade_history
from ui.pages.settings import show_settings


# Page configuration
st.set_page_config(
    page_title="Stock Analyzer V15",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    
    # Sidebar navigation
    st.sidebar.title("📈 Stock Analyzer V15")
    st.sidebar.markdown("---")
    
    # Navigation menu
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Stock Analysis", "Portfolio", "Trade History", "Settings"],
        label_visibility="collapsed"
    )
    
    # Display selected page
    if page == "Dashboard":
        show_dashboard()
    elif page == "Stock Analysis":
        show_stock_analysis()
    elif page == "Portfolio":
        show_portfolio()
    elif page == "Trade History":
        show_trade_history()
    elif page == "Settings":
        show_settings()


if __name__ == "__main__":
    main()


