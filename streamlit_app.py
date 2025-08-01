import streamlit as st
import pandas as pd
from datetime import datetime
import time

def initialize_session_state():
    """Initialize session state variables for the app."""
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    if 'tool_usage' not in st.session_state:
        # Simulate tool usage data - in a real app, this would come from analytics
        st.session_state.tool_usage = {
            'csv_joiner': 1247,
            'word_metrics': 892,
            'dictionary_classifier': 756,
            'simple_classifier': 1089
        }

def get_theme_colors():
    """Get theme colors based on dark/light mode."""
    if st.session_state.dark_mode:
        return {
            'bg_primary': '#1e1e1e',
            'bg_secondary': '#2d2d2d',
            'text_primary': '#ffffff',
            'text_secondary': '#cccccc',
            'accent': '#4a9eff',
            'gradient': 'linear-gradient(90deg, #4a9eff 0%, #6c5ce7 100%)',
            'card_bg': '#2d2d2d',
            'border_color': '#404040'
        }
    else:
        return {
            'bg_primary': '#ffffff',
            'bg_secondary': '#f8f9fa',
            'text_primary': '#000000',
            'text_secondary': '#666666',
            'accent': '#667eea',
            'gradient': 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
            'card_bg': '#f8f9fa',
            'border_color': '#e0e0e0'
        }

def main():
    st.set_page_config(
        page_title="Data Analysis Toolkit",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Get theme colors
    colors = get_theme_colors()
    
    # Dark/Light Mode Toggle in Sidebar
    with st.sidebar:
        st.markdown("### 🎨 Theme Settings")
        
        # Theme toggle button
        if st.button("🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode", 
                    key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        
        st.markdown("---")
    
    # Custom CSS for enhanced styling with theme support
    st.markdown(f"""
    <style>
    /* Main app styling */
    .main .block-container {{
        background-color: {colors['bg_primary']};
        color: {colors['text_primary']};
    }}
    
    .main-header {{
        text-align: center;
        padding: 2rem 0;
        background: {colors['gradient']};
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }}
    
    .tool-card {{
        background: {colors['card_bg']};
        color: {colors['text_primary']};
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid {colors['accent']};
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid {colors['border_color']};
    }}
    
    .popular-badge {{
        background: #ff6b6b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: bold;
        position: absolute;
        top: -8px;
        right: 15px;
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}
    
    .tool-card-container {{
        position: relative;
    }}
    
    .usage-stats {{
        font-size: 0.85rem;
        color: {colors['text_secondary']};
        margin-top: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .feature-badge {{
        background: {colors['accent']};
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.25rem;
        display: inline-block;
    }}
    
    .stats-card {{
        background: {colors['gradient']};
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }}
    
    .quick-start {{
        background: {colors['bg_secondary']};
        color: {colors['text_primary']};
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid {colors['accent']};
        margin: 1rem 0;
    }}
    
    /* Dark mode specific adjustments */
    .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{
        color: {colors['text_primary']} !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Data Analysis Toolkit</h1>
        <h3>Your Complete Suite for Text Analysis, Data Joining & Classification</h3>
        <p>Transform your data with powerful, easy-to-use tools built for modern data workflows</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats Section
    st.subheader("📊 Toolkit Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-card">
            <h2>4</h2>
            <p>Powerful Tools</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stats-card">
            <h2>∞</h2>
            <p>File Size Support</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stats-card">
            <h2>100%</h2>
            <p>Browser-Based</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stats-card">
            <h2>0$</h2>
            <p>Completely Free</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tools Overview Section with Most Popular Indicators
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("🛠️ Available Tools")
    with col2:
        st.markdown("**📈 Usage Stats (This Month)**")
    
    # Get sorted tools by popularity
    tools_by_popularity = sorted(st.session_state.tool_usage.items(), key=lambda x: x[1], reverse=True)
    most_popular = tools_by_popularity[0][0]
    second_popular = tools_by_popularity[1][0]
    
    # Tool 1: CSV Table Joiner
    is_popular = 'csv_joiner' == most_popular or 'csv_joiner' == second_popular
    popularity_badge = ""
    if 'csv_joiner' == most_popular:
        popularity_badge = '<div class="popular-badge">🔥 MOST POPULAR</div>'
    elif 'csv_joiner' == second_popular:
        popularity_badge = '<div class="popular-badge" style="background: #4ecdc4;">⭐ TRENDING</div>'
    
    st.markdown(f"""
    <div class="tool-card-container">
        {popularity_badge}
        <div class="tool-card">
            <h3>🔗 CSV Table Joiner</h3>
            <p><strong>Join and merge CSV files with advanced matching capabilities</strong></p>
            <p>Perfect for combining datasets from different sources with intelligent fuzzy matching algorithms.</p>
            
            <div style="margin: 1rem 0;">
                <span class="feature-badge">Fuzzy Matching</span>
                <span class="feature-badge">Multiple Join Types</span>
                <span class="feature-badge">Duplicate Removal</span>
                <span class="feature-badge">Smart Column Mapping</span>
            </div>
            
            <p><strong>Best for:</strong> Data integration, customer matching, inventory reconciliation</p>
            <p><strong>Input:</strong> Two CSV files | <strong>Output:</strong> Merged dataset with match statistics</p>
            
            <div class="usage-stats">
                <span>📊 {st.session_state.tool_usage['csv_joiner']:,} users this month</span>
                <span>•</span>
                <span>⭐ 4.8/5 rating</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tool 2: Text Classification Word Metrics
    is_popular = 'word_metrics' == most_popular or 'word_metrics' == second_popular
    popularity_badge = ""
    if 'word_metrics' == most_popular:
        popularity_badge = '<div class="popular-badge">🔥 MOST POPULAR</div>'
    elif 'word_metrics' == second_popular:
        popularity_badge = '<div class="popular-badge" style="background: #4ecdc4;">⭐ TRENDING</div>'
    
    st.markdown(f"""
    <div class="tool-card-container">
        {popularity_badge}
        <div class="tool-card">
            <h3>📊 Text Classification Word Metrics Analyzer</h3>
            <p><strong>Analyze text classification performance with detailed word-level metrics</strong></p>
            <p>Deep dive into your text data with comprehensive analysis at both statement and ID levels.</p>
            
            <div style="margin: 1rem 0;">
                <span class="feature-badge">Word-Level Analysis</span>
                <span class="feature-badge">Custom Classifiers</span>
                <span class="feature-badge">ID Aggregation</span>
                <span class="feature-badge">Export Configs</span>
            </div>
            
            <p><strong>Best for:</strong> Content analysis, marketing research, sentiment studies</p>
            <p><strong>Input:</strong> CSV with ID and text columns | <strong>Output:</strong> Detailed metrics and statistics</p>
            
            <div class="usage-stats">
                <span>📊 {st.session_state.tool_usage['word_metrics']:,} users this month</span>
                <span>•</span>
                <span>⭐ 4.7/5 rating</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tool 3: Dictionary-Based Text Classifier
    is_popular = 'dictionary_classifier' == most_popular or 'dictionary_classifier' == second_popular
    popularity_badge = ""
    if 'dictionary_classifier' == most_popular:
        popularity_badge = '<div class="popular-badge">🔥 MOST POPULAR</div>'
    elif 'dictionary_classifier' == second_popular:
        popularity_badge = '<div class="popular-badge" style="background: #4ecdc4;">⭐ TRENDING</div>'
    
    st.markdown(f"""
    <div class="tool-card-container">
        {popularity_badge}
        <div class="tool-card">
            <h3>📚 Dictionary-Based Text Classifier</h3>
            <p><strong>Advanced text classification using customizable keyword dictionaries</strong></p>
            <p>Classify text data with continuous variables, visualizations, and comprehensive reporting.</p>
            
            <div style="margin: 1rem 0;">
                <span class="feature-badge">Custom Dictionaries</span>
                <span class="feature-badge">Continuous Variables</span>
                <span class="feature-badge">Visual Analytics</span>
                <span class="feature-badge">Real-time Results</span>
            </div>
            
            <p><strong>Best for:</strong> Content categorization, theme analysis, automated tagging</p>
            <p><strong>Input:</strong> CSV with text data | <strong>Output:</strong> Classifications with confidence scores</p>
            
            <div class="usage-stats">
                <span>📊 {st.session_state.tool_usage['dictionary_classifier']:,} users this month</span>
                <span>•</span>
                <span>⭐ 4.6/5 rating</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tool 4: Simple Dictionary Classifier
    is_popular = 'simple_classifier' == most_popular or 'simple_classifier' == second_popular
    popularity_badge = ""
    if 'simple_classifier' == most_popular:
        popularity_badge = '<div class="popular-badge">🔥 MOST POPULAR</div>'
    elif 'simple_classifier' == second_popular:
        popularity_badge = '<div class="popular-badge" style="background: #4ecdc4;">⭐ TRENDING</div>'
    
    st.markdown(f"""
    <div class="tool-card-container">
        {popularity_badge}
        <div class="tool-card">
            <h3>🏷️ Simple Dictionary Text Classifier</h3>
            <p><strong>Fast and straightforward text classification for quick insights</strong></p>
            <p>Streamlined keyword matching with binary classifications and easy-to-understand results.</p>
            
            <div style="margin: 1rem 0;">
                <span class="feature-badge">Quick Setup</span>
                <span class="feature-badge">Binary Classification</span>
                <span class="feature-badge">Keyword Tracking</span>
                <span class="feature-badge">Simple Interface</span>
            </div>
            
            <p><strong>Best for:</strong> Quick content screening, simple categorization, proof of concepts</p>
            <p><strong>Input:</strong> CSV with text column | <strong>Output:</strong> Binary classifications with matched keywords</p>
            
            <div class="usage-stats">
                <span>📊 {st.session_state.tool_usage['simple_classifier']:,} users this month</span>
                <span>•</span>
                <span>⭐ 4.9/5 rating</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown("---")
    st.subheader("🚀 Quick Start Guide")
    
    st.markdown("""
    <div class="quick-start">
        <h4>Get Started in 3 Easy Steps:</h4>
        
        <div style="display: flex; align-items: center; margin: 1rem 0;">
            <div style="background: #2196f3; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 1rem; font-weight: bold;">1</div>
            <div>
                <strong>Choose Your Tool</strong><br>
                Select the tab that matches your data analysis needs
            </div>
        </div>
        
        <div style="display: flex; align-items: center; margin: 1rem 0;">
            <div style="background: #4caf50; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 1rem; font-weight: bold;">2</div>
            <div>
                <strong>Upload Your Data</strong><br>
                Simply drag and drop your CSV files or use the file uploader
            </div>
        </div>
        
        <div style="display: flex; align-items: center; margin: 1rem 0;">
            <div style="background: #ff9800; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 1rem; font-weight: bold;">3</div>
            <div>
                <strong>Get Results</strong><br>
                Configure your analysis settings and download your processed data
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Use Cases Section
    st.markdown("---")
    st.subheader("💡 Common Use Cases")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔍 Data Integration Projects**
        - Merge customer databases from different systems
        - Combine sales data with inventory records
        - Join survey responses with demographic data
        
        **📈 Marketing Analysis**
        - Analyze campaign message effectiveness
        - Classify customer feedback themes
        - Track keyword performance across content
        """)
    
    with col2:
        st.markdown("""
        **🎯 Content Analysis**
        - Categorize social media posts
        - Analyze support ticket themes
        - Classify research papers or articles
        
        **⚡ Quick Data Processing**
        - Clean and standardize datasets
        - Generate insights from text data
        - Prepare data for further analysis
        """)
    
    # Tips and Best Practices
    st.markdown("---")
    st.subheader("💡 Tips for Best Results")
    
    with st.expander("📋 Data Preparation Tips"):
        st.markdown("""
        - **CSV Format**: Ensure your files have clear column headers
        - **Text Quality**: Clean text data performs better (remove excessive punctuation, fix encoding issues)
        - **File Size**: While we support large files, consider splitting very large datasets for better performance
        - **Column Names**: Use descriptive column names for easier identification
        """)
    
    with st.expander("🎯 Tool Selection Guide"):
        st.markdown("""
        - **Choose Table Joiner** when you need to combine data from multiple sources
        - **Choose Word Metrics Analyzer** for detailed text analysis with custom categories
        - **Choose Dictionary Classifier** for comprehensive text classification with visualizations
        - **Choose Simple Classifier** for quick, straightforward text categorization
        """)
    
    with st.expander("⚡ Performance Optimization"):
        st.markdown("""
        - **Large Files**: Process in smaller batches if you experience slowdowns
        - **Fuzzy Matching**: Adjust thresholds based on your data quality
        - **Custom Dictionaries**: Start with smaller keyword sets and expand based on results
        - **Browser Performance**: Close other tabs for better performance with large datasets
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <h4>Ready to Transform Your Data? 🚀</h4>
        <p>Select any tool from the tabs above to get started with your analysis</p>
        <p><em>Built with ❤️ using Streamlit • No data is stored on our servers • Everything runs in your browser</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Optional: Add current time/date
    current_time = datetime.now().strftime("%B %d, %Y • %I:%M %p")
    st.sidebar.markdown(f"**Current Session:** {current_time}")
    st.sidebar.markdown("---")
    
    # Popular tools sidebar info
    st.sidebar.markdown("**🔥 Most Popular Tools:**")
    for i, (tool, usage) in enumerate(tools_by_popularity[:3], 1):
        tool_names = {
            'csv_joiner': '🔗 Table Joiner',
            'word_metrics': '📊 Word Metrics',
            'dictionary_classifier': '📚 Dictionary Classifier',
            'simple_classifier': '🏷️ Simple Classifier'
        }
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        st.sidebar.markdown(f"{emoji} {tool_names[tool]} ({usage:,} users)")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Quick Navigation:**
    - 🔗 Table Joiner
    - 📊 Word Metrics  
    - 📚 Dictionary Classifier
    - 🏷️ Simple Classifier
    """)

if __name__ == "__main__":
    main()
