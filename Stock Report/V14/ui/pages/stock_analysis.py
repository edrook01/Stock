"""
Stock Analysis Page
Comprehensive stock analysis with predictions, charts, and technical indicators.
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.data_fetcher import fetch_prices
from core.indicators import rsi, sma, ema
from core.timeframes import CFD_TIMEFRAMES, INVESTMENT_TIMEFRAMES, ALL_TIMEFRAMES
from model.unified_model import get_model
from model.feature_extractor import FeatureExtractor
from risk.volatility import calculate_atr
from risk.stop_loss import calculate_stop_loss_distance, calculate_stop_loss_price
from risk.profiles import get_risk_profile, RiskProfile
from sentiment.override import get_sentiment_override


def show_stock_analysis():
    """Display stock analysis page."""
    
    st.title("📈 Stock Analysis")
    st.markdown("---")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input(
            "Ticker Symbol",
            value="AAPL",
            placeholder="Enter ticker (e.g., AAPL, TSLA)",
            help="Enter a valid stock ticker symbol"
        ).upper().strip()
    
    with col2:
        timeframe = st.selectbox(
            "Timeframe",
            options=ALL_TIMEFRAMES,
            index=ALL_TIMEFRAMES.index("1d") if "1d" in ALL_TIMEFRAMES else 0,
            help="Select prediction timeframe"
        )
    
    if not ticker:
        st.info("Please enter a ticker symbol to begin analysis.")
        return
    
    # Analyze button
    if st.button("Analyze", type="primary", use_container_width=True):
        _analyze_stock(ticker, timeframe)
    else:
        # Show placeholder when not analyzing
        st.info("Click 'Analyze' to generate stock analysis and predictions.")


def _analyze_stock(ticker: str, timeframe: str):
    """Perform comprehensive stock analysis."""
    
    with st.spinner(f"Fetching data and generating prediction for {ticker} ({timeframe})..."):
        try:
            # Fetch price data (async)
            df = asyncio.run(fetch_prices(ticker, timeframe))
            
            if df is None or df.empty:
                st.error(f"❌ Could not fetch price data for {ticker}. Please check the ticker symbol and try again.")
                return
            
            # Validate DataFrame has required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                return
            
            # Check data sufficiency
            min_rows_needed = 50  # For most indicators
            if len(df) < min_rows_needed:
                st.warning(f"⚠️ Limited data available ({len(df)} rows). Some indicators may not be available.")
            
            # Display price chart
            _display_price_chart(df, ticker)
            
            st.markdown("---")
            
            # Get prediction (async)
            prediction_result = _get_prediction(ticker, timeframe, df)
            
            # Display prediction
            _display_prediction(prediction_result, ticker, timeframe)
            
            st.markdown("---")
            
            # Technical indicators
            _display_technical_indicators(df, ticker)
            
            st.markdown("---")
            
            # Feature extraction display
            _display_features(df, ticker, timeframe)
            
            st.markdown("---")
            
            # Risk assessment
            _display_risk_assessment(df, ticker, prediction_result)
            
            st.markdown("---")
            
            # Sentiment override status
            _display_sentiment_status(ticker)
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)


def _display_price_chart(df: pd.DataFrame, ticker: str):
    """Display price chart."""
    st.subheader(f"📊 Price Chart - {ticker}")
    
    if df is None or df.empty:
        st.info("No price data available")
        return
    
    # Prepare chart data
    chart_df = df[['Close']].copy()
    chart_df.columns = ['Price']
    
    # Display line chart
    st.line_chart(chart_df)
    
    # Show current price info
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = float(df['Close'].iloc[-1])
    prev_price = float(df['Close'].iloc[-2]) if len(df) >= 2 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price * 100) if prev_price > 0 else 0
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        st.metric("Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
    
    with col3:
        high_52w = float(df['High'].max()) if len(df) > 0 else current_price
        st.metric("Period High", f"${high_52w:.2f}")
    
    with col4:
        low_52w = float(df['Low'].min()) if len(df) > 0 else current_price
        st.metric("Period Low", f"${low_52w:.2f}")


def _get_prediction(ticker: str, timeframe: str, df: pd.DataFrame) -> dict:
    """Get prediction from unified model."""
    try:
        model = get_model(timeframe)
        prediction = asyncio.run(model.predict(ticker, df=df))
        
        # Validate prediction result
        if not prediction or not isinstance(prediction, dict):
            return _default_prediction_dict(timeframe, ticker)
        
        # Ensure all required keys exist
        required_keys = ['prediction', 'confidence', 'range_low', 'range_high', 'timeframe']
        for key in required_keys:
            if key not in prediction:
                return _default_prediction_dict(timeframe, ticker)
        
        return prediction
        
    except Exception as e:
        st.warning(f"⚠️ Error generating prediction: {str(e)}")
        return _default_prediction_dict(timeframe, ticker)


def _default_prediction_dict(timeframe: str, ticker: str) -> dict:
    """Return default prediction when model not trained or error occurs."""
    return {
        "prediction": 0.0,
        "confidence": 0.5,
        "range_low": -1.0,
        "range_high": 1.0,
        "timeframe": timeframe,
        "ticker": ticker,
        "model_agreement": 0.0,
        "is_default": True
    }


def _display_prediction(prediction: dict, ticker: str, timeframe: str):
    """Display prediction results."""
    st.subheader("🤖 ML Model Prediction")
    
    pred_value = prediction.get('prediction', 0.0)
    confidence = prediction.get('confidence', 0.5)
    range_low = prediction.get('range_low', -1.0)
    range_high = prediction.get('range_high', 1.0)
    is_default = prediction.get('is_default', False)
    
    # Check if model is trained
    model = get_model(timeframe)
    if not model.is_trained or is_default:
        st.warning("⚠️ Model not trained for this timeframe. Showing default prediction.")
        st.info("Train the model using historical trade data to get accurate predictions.")
    
    # Display prediction metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Prediction value
        color = "green" if pred_value > 0 else "red" if pred_value < 0 else "gray"
        st.metric(
            "Predicted Movement",
            f"{pred_value:+.2f}%",
            delta=f"{range_low:.2f}% to {range_high:.2f}%"
        )
    
    with col2:
        # Confidence with visual indicator
        confidence_pct = confidence * 100
        st.metric("Confidence", f"{confidence_pct:.1f}%")
        st.progress(confidence)
    
    with col3:
        # Prediction range
        st.metric("Prediction Range", f"{range_low:.2f}% to {range_high:.2f}%")
    
    # Model agreement if available
    if 'model_agreement' in prediction:
        agreement = prediction.get('model_agreement', 0.0)
        st.caption(f"Model Agreement: {agreement:.1%}")


def _display_technical_indicators(df: pd.DataFrame, ticker: str):
    """Display technical indicators."""
    st.subheader("📊 Technical Indicators")
    
    if df is None or df.empty:
        st.info("No data available for indicators")
        return
    
    close = df['Close']
    
    # RSI
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**RSI (14)**")
        try:
            if len(df) >= 15:
                rsi_value = rsi(close, period=14)
                st.metric("RSI", f"{rsi_value:.2f}")
                
                # RSI interpretation
                if rsi_value > 70:
                    st.caption("🔴 Overbought")
                elif rsi_value < 30:
                    st.caption("🟢 Oversold")
                else:
                    st.caption("⚪ Neutral")
            else:
                st.info("Need ≥15 data points")
        except Exception as e:
            st.error(f"Error calculating RSI: {e}")
    
    with col2:
        st.write("**Moving Averages**")
        try:
            if len(df) >= 20:
                sma20 = sma(close, period=20)
                ema20 = ema(close, period=20)
                
                current_price = float(close.iloc[-1])
                
                st.metric("SMA(20)", f"${sma20:.2f}")
                st.metric("EMA(20)", f"${ema20:.2f}")
                
                # Price vs MA
                price_vs_sma = ((current_price / sma20) - 1) * 100
                st.caption(f"Price vs SMA(20): {price_vs_sma:+.2f}%")
            else:
                st.info("Need ≥20 data points")
        except Exception as e:
            st.error(f"Error calculating MAs: {e}")
    
    with col3:
        st.write("**Volatility (ATR)**")
        try:
            if len(df) >= 15:
                atr_value = calculate_atr(df, period=14)
                current_price = float(close.iloc[-1])
                atr_pct = (atr_value / current_price * 100) if current_price > 0 else 0
                
                st.metric("ATR(14)", f"${atr_value:.2f}")
                st.caption(f"ATR as % of price: {atr_pct:.2f}%")
            else:
                st.info("Need ≥15 data points")
        except Exception as e:
            st.error(f"Error calculating ATR: {e}")


def _display_features(df: pd.DataFrame, ticker: str, timeframe: str):
    """Display extracted features."""
    st.subheader("🔍 Extracted Features")
    
    try:
        feature_extractor = FeatureExtractor()
        features = asyncio.run(feature_extractor.extract_features(
            ticker=ticker,
            interval=timeframe,
            df=df
        ))
        
        if not features:
            st.info("No features extracted. Insufficient data.")
            return
        
        # Group features by category
        price_features = {k: v for k, v in features.items() if 'price' in k.lower()}
        volume_features = {k: v for k, v in features.items() if 'volume' in k.lower()}
        volatility_features = {k: v for k, v in features.items() if 'volatility' in k.lower()}
        momentum_features = {k: v for k, v in features.items() if 'momentum' in k.lower()}
        technical_features = {k: v for k, v in features.items() 
                             if k.startswith(('rsi', 'sma', 'ema'))}
        
        # Display in columns
        col1, col2 = st.columns(2)
        
        with col1:
            if price_features:
                st.write("**Price Features**")
                for key, value in price_features.items():
                    if isinstance(value, (int, float)):
                        st.text(f"{key}: {value:.4f}")
            
            if volume_features:
                st.write("**Volume Features**")
                for key, value in volume_features.items():
                    if isinstance(value, (int, float)):
                        st.text(f"{key}: {value:.2f}")
        
        with col2:
            if volatility_features:
                st.write("**Volatility Features**")
                for key, value in volatility_features.items():
                    if isinstance(value, (int, float)):
                        st.text(f"{key}: {value:.4f}")
            
            if momentum_features:
                st.write("**Momentum Features**")
                for key, value in momentum_features.items():
                    if isinstance(value, (int, float)):
                        st.text(f"{key}: {value:.4f}")
        
        # Technical indicators
        if technical_features:
            st.write("**Technical Indicators**")
            tech_cols = st.columns(min(len(technical_features), 4))
            for idx, (key, value) in enumerate(technical_features.items()):
                if isinstance(value, (int, float)):
                    with tech_cols[idx % len(tech_cols)]:
                        st.metric(key.upper(), f"{value:.2f}")
        
    except Exception as e:
        st.warning(f"⚠️ Error extracting features: {str(e)}")


def _display_risk_assessment(df: pd.DataFrame, ticker: str, prediction: dict):
    """Display risk assessment with ATR-based stop-loss preview."""
    st.subheader("⚠️ Risk Assessment")
    
    if df is None or df.empty:
        st.info("No data available for risk assessment")
        return
    
    try:
        current_price = float(df['Close'].iloc[-1])
        confidence = prediction.get('confidence', 0.5)
        # Get risk profile from config
        from core.portable_paths import get_data_path
        import json
        try:
            config_file = get_data_path() / 'config_v14.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                profile_name = config.get("risk_profile", "medium")
                risk_profile = get_risk_profile(profile_name) or RiskProfile.MEDIUM
            else:
                risk_profile = RiskProfile.MEDIUM
        except:
            risk_profile = RiskProfile.MEDIUM
        
        # Calculate ATR
        try:
            if len(df) >= 15:
                atr_value = calculate_atr(df, period=14)
                
                # Calculate stop-loss distance
                stop_distance, atr_used = calculate_stop_loss_distance(
                    df=df,
                    profile=risk_profile,
                    confidence=confidence,
                    asset_risk_category="medium"  # Could be enhanced with asset classification
                )
                
                # Calculate stop prices for LONG and SHORT
                stop_long = calculate_stop_loss_price(
                    entry_price=current_price,
                    direction="LONG",
                    stop_distance=stop_distance
                )
                
                stop_short = calculate_stop_loss_price(
                    entry_price=current_price,
                    direction="SHORT",
                    stop_distance=stop_distance
                )
                
                # Display risk metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("ATR (14)", f"${atr_value:.2f}")
                    st.caption(f"ATR %: {(atr_value/current_price*100):.2f}%")
                
                with col2:
                    st.metric("Stop Distance", f"${stop_distance:.2f}")
                    st.caption(f"Stop %: {(stop_distance/current_price*100):.2f}%")
                
                with col3:
                    st.metric("Risk Profile", risk_profile.value.upper())
                    st.caption(f"Confidence: {confidence:.1%}")
                
                # Stop-loss prices
                st.write("**Stop-Loss Prices**")
                stop_col1, stop_col2 = st.columns(2)
                
                with stop_col1:
                    st.write("**LONG Position**")
                    st.metric("Entry", f"${current_price:.2f}")
                    st.metric("Stop-Loss", f"${stop_long:.2f}")
                    loss_pct_long = ((current_price - stop_long) / current_price * 100)
                    st.caption(f"Risk: {loss_pct_long:.2f}%")
                
                with stop_col2:
                    st.write("**SHORT Position**")
                    st.metric("Entry", f"${current_price:.2f}")
                    st.metric("Stop-Loss", f"${stop_short:.2f}")
                    loss_pct_short = ((stop_short - current_price) / current_price * 100)
                    st.caption(f"Risk: {loss_pct_short:.2f}%")
                
            else:
                st.warning("⚠️ Insufficient data for ATR calculation (need ≥15 data points)")
        
        except ValueError as e:
            st.warning(f"⚠️ Could not calculate ATR: {str(e)}")
        except Exception as e:
            st.warning(f"⚠️ Error in risk assessment: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Error displaying risk assessment: {str(e)}")


def _display_sentiment_status(ticker: str):
    """Display sentiment override status."""
    st.subheader("📰 Sentiment Override Status")
    
    try:
        sentiment_override = get_sentiment_override()
        status = sentiment_override.get_override_status()
        
        # Check if trade would be blocked
        should_block, reason = sentiment_override.should_block_trade(ticker)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Override Status**")
            if status.get('protective_mode', False):
                st.warning("🛡️ Protective Mode: ACTIVE")
            else:
                st.success("✅ Protective Mode: Inactive")
            
            blocked_count = len(status.get('blocked_tickers', {}))
            st.metric("Blocked Tickers", blocked_count)
        
        with col2:
            st.write("**Trade Status**")
            if should_block:
                st.error(f"🚫 Trade Blocked: {reason}")
            else:
                st.success("✅ Trade Allowed")
            
            threshold = status.get('override_threshold', 0.7)
            st.caption(f"Override Threshold: {threshold:.2f}")
    
    except Exception as e:
        st.warning(f"⚠️ Could not load sentiment status: {str(e)}")

