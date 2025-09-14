import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from lottery_predictor import LotterySystem
import tempfile
import os
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import threading
import time

# Page configuration
st.set_page_config(
    page_title="🎲 Lottery Number Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for lottery card styling
st.markdown("""
<style>
/* Lottery card styling to match the image */
.lottery-card {
    background: #1e3c72;
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
    color: white;
    font-family: Arial, sans-serif;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.lottery-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    font-weight: bold;
}

.lottery-date {
    font-size: 1.1em;
}

.lottery-id {
    font-size: 1em;
    font-weight: normal;
}

.lottery-time {
    font-size: 1.1em;
}

.lottery-numbers {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 15px;
    flex-wrap: wrap;
}

.number-circle {
    width: 35px;
    height: 35px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 0.9em;
}

.number-white {
    background: white;
    color: #1e3c72;
}

.number-yellow {
    background: #ffd700;
    color: #1e3c72;
}

.plus-sign {
    color: white;
    font-size: 1.2em;
    font-weight: bold;
    margin: 0 5px;
}

.money-dots {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-left: 10px;
}

.money-dots-text {
    color: white;
    font-weight: bold;
    font-size: 0.9em;
    margin-left: 5px;
}

.extra-text {
    color: white;
    font-size: 0.8em;
    margin-top: 5px;
    text-align: right;
}

/* Training output styling */
.training-output {
    background: #f0f2f6;
    border: 1px solid #e1e5e9;
    border-radius: 8px;
    padding: 15px;
    margin: 15px 0;
    font-family: monospace;
    font-size: 0.9em;
    max-height: 300px;
    overflow-y: auto;
}

/* Hide streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}
.stDecoration {display:none;}
.stStatusWidget {display:none;}
.stProgressLabel {display:none;}
.lottery-card {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    border-radius: 20px;
    padding: 25px;
    margin: 20px 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    border: 2px solid #4a90e2;
}

.lottery-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    color: white;
}

.lottery-date {
    font-size: 18px;
    font-weight: bold;
}

.lottery-time {
    font-size: 16px;
    font-weight: bold;
}

.lottery-numbers {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 20px 0;
    justify-content: center;
}

.number-circle {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 16px;
    color: black;
    border: 2px solid #ddd;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.money-dots {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 20px 0;
    justify-content: center;
}

.money-circle {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: #ffd700;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 16px;
    color: black;
    border: 2px solid #ffed4e;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.money-label {
    color: white;
    font-size: 18px;
    font-weight: bold;
    text-transform: uppercase;
}

.plus-sign {
    color: white;
    font-size: 24px;
    font-weight: bold;
    margin: 0 10px;
}

.draw-title {
    color: white;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 15px;
}

.stats-container {
    background: rgba(255,255,255,0.1);
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
    color: white;
}

.stats-title {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 15px;
    text-align: center;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
}

.stat-item {
    text-align: center;
    padding: 10px;
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
}

.stat-value {
    font-size: 24px;
    font-weight: bold;
    color: #ffd700;
}

.stat-label {
    font-size: 14px;
    margin-top: 5px;
}

.upload-section {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
    border: 2px dashed #4a90e2;
}

.upload-title {
    color: white;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 15px;
}

.success-message {
    background: rgba(76, 175, 80, 0.2);
    border: 1px solid #4caf50;
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
    color: #4caf50;
    text-align: center;
}

.error-message {
    background: rgba(244, 67, 54, 0.2);
    border: 1px solid #f44336;
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
    color: #f44336;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

def display_lottery_draws_circles(predictions):
    """Display lottery draws as white circular buttons with dark blue text."""
    for i, numbers in enumerate(predictions, 1):
        st.write(f"**Draw {i}:**")
        
        # Create columns for the circular numbers
        cols = st.columns(len(numbers))
        
        for j, num in enumerate(numbers):
            with cols[j]:
                # Create circular button style
                st.markdown(f"""
                <div style="
                    width: 40px; 
                    height: 40px; 
                    border-radius: 50%; 
                    background-color: white; 
                    border: 2px solid #1e3c72;
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    margin: 5px;
                    font-weight: bold;
                    font-size: 14px;
                    color: #1e3c72;
                ">
                    {num:02d}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

def display_stats(predictions):
    """Display statistics in a nice format."""
    all_numbers = [num for draw in predictions for num in draw]
    
    # Calculate statistics
    most_frequent = max(set(all_numbers), key=all_numbers.count)
    number_range = f"{min(all_numbers)}-{max(all_numbers)}"
    avg_per_draw = np.mean([np.mean(draw) for draw in predictions])
    
    # Count frequency of each number
    freq_count = {}
    for num in all_numbers:
        freq_count[num] = freq_count.get(num, 0) + 1
    
    # Get top 5 most frequent
    top_frequent = sorted(freq_count.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Display statistics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Draws", len(predictions))
    with col2:
        st.metric("Number Range", number_range)
    with col3:
        st.metric("Most Frequent", most_frequent)
    with col4:
        st.metric("Average", f"{avg_per_draw:.1f}")
    
    # Display top numbers in a nice format
    st.write("**Top 5 Most Predicted Numbers:**")
    cols = st.columns(5)
    for i, (num, count) in enumerate(top_frequent):
        with cols[i]:
            st.metric(f"#{num}", f"{count}x")

class TerminalOutput:
    def __init__(self):
        self.content = []
        self.max_lines = 50
    
    def write(self, text):
        if text.strip():
            self.content.append(text.strip())
            if len(self.content) > self.max_lines:
                self.content.pop(0)
    
    def flush(self):
        pass
    
    def get_content(self):
        return '\n'.join(self.content)

def capture_training_output(system, tmp_path, num_predictions, progress_bar, status_text, terminal_output, show_training):
    """Capture training output and update UI"""
    try:
        if show_training:
            # Redirect stdout to capture training output
            with redirect_stdout(terminal_output):
                with redirect_stderr(terminal_output):
                    # Initialize system
                    status_text.text("🔄 Initializing prediction system...")
                    progress_bar.progress(10)
                    time.sleep(0.5)
                    
                    # Train model
                    status_text.text("🚀 Training AI model with optimal settings...")
                    progress_bar.progress(30)
                    
                    features_df = system.train(tmp_path, sequence_length=50, prediction_length=num_predictions)
                    
                    # Generate predictions
                    status_text.text("🔮 Generating lottery numbers...")
                    progress_bar.progress(80)
                    
                    predictions = system.predict(features_df, sequence_length=50)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Predictions generated successfully!")
                    
                    return predictions, features_df
        else:
            # Initialize system
            status_text.text("🔄 Initializing prediction system...")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            # Train model
            status_text.text("🚀 Training AI model with optimal settings...")
            progress_bar.progress(30)
            
            features_df = system.train(tmp_path, sequence_length=50, prediction_length=num_predictions)
            
            # Generate predictions
            status_text.text("🔮 Generating lottery numbers...")
            progress_bar.progress(80)
            
            predictions = system.predict(features_df, sequence_length=50)
            
            progress_bar.progress(100)
            status_text.text("✅ Predictions generated successfully!")
            
            return predictions, features_df
                
    except Exception as e:
        if show_training:
            terminal_output.write(f"ERROR: {str(e)}")
        raise e

def main():
    # Simple header
    st.title("🎲 Lottery Number Predictor")
    st.markdown("AI-Powered Lottery Number Generation")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your lottery data CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: 'Draw Date' and 'Winning Numbers'"
    )
    
    # Settings
    col1, col2 = st.columns(2)
    with col1:
        num_predictions = st.slider("Number of Predictions", min_value=1, max_value=10, value=5, step=1)
    with col2:
        show_training = st.checkbox("Show Training Output", value=True)
        
    # Predict button
    if st.button("🎯 Generate Predictions", type="primary", use_container_width=True):
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Create containers for live updates
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize terminal output capture
                terminal_output = TerminalOutput()
                system = LotterySystem()
                
                # Show training output if requested
                if show_training:
                    st.markdown("### 🖥️ Training Output")
                    terminal_container = st.empty()
                
                # Start training with live output
                predictions, features_df = capture_training_output(
                    system, tmp_path, num_predictions, progress_bar, status_text, terminal_output, show_training
                )
                
                # Display final terminal output if requested
                if show_training:
                    st.code(terminal_output.get_content(), language="text")
                
                # Generate and display analysis plot
                st.markdown("---")
                st.markdown("## 📊 Data Analysis")
                
                # Load original data for analysis
                uploaded_file.seek(0)  # Reset file pointer
                raw_df = pd.read_csv(uploaded_file)
                raw_df['Draw Date'] = pd.to_datetime(raw_df['Draw Date'])
                
                # Create and display analysis plot
                analysis_fig = system.analyze_and_visualize(raw_df, predictions)
                st.pyplot(analysis_fig)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                # Display results
                st.markdown("---")
                st.markdown("## 🎲 Your Lottery Predictions")
                
                # Display lottery draws as circular buttons
                display_lottery_draws_circles(predictions)
                
                # Add statistics
                display_stats(predictions)
                
                # Success message
                st.success("🎉 Predictions generated successfully! Good luck with your lottery numbers!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please upload a CSV file first.")
    
    st.markdown("---")
    st.write("**How to Use:**")
    st.write("1. Upload your lottery data CSV file")
    st.write("2. Choose number of predictions")
    st.write("3. Click 'Generate Predictions' to train the AI model")
    st.write("4. View your lottery number predictions")
    
    st.write("**Data Format:** Your CSV should have columns: `Draw Date` and `Winning Numbers`")
    
    # Remove AI model name from display
    st.write("**Model:** Advanced AI System (Proprietary)")
    
    st.markdown("---")
    st.write("*AI-Powered Lottery Prediction System | Use responsibly and for entertainment only*")

if __name__ == "__main__":
    main()
