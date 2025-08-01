import streamlit as st
import pandas as pd
from datetime import datetime
import time

def main():
    st.set_page_config(
        page_title="Data Analysis Toolkit",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .tool-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .feature-badge {
        background: #667eea;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.25rem;
        display: inline-block;
    }
    .popular-indicator {
        background: #ff4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-left: 1rem;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .quick-start {
        background: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #2196f3;
        margin: 1rem 0;
    }
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
    
    # Tools Overview Section
    st.markdown("---")
    st.subheader("🛠️ Available Tools")
    
    # Tool 1: Join Table App
    st.markdown("### 🔗 Join Table App 🔥 **MOST POPULAR**")
    st.markdown("**Join and merge CSV files with advanced matching capabilities**")
    st.markdown("Perfect for combining datasets from different sources with intelligent fuzzy matching algorithms.")
    st.markdown("**Key Features:** Fuzzy Matching • Multiple Join Types • Duplicate Removal • Smart Column Mapping")
    st.markdown("**Best for:** Data integration, customer matching, inventory reconciliation")
    st.markdown("**Input:** Two CSV files | **Output:** Merged dataset with match statistics")
    st.markdown("")
    
    # Tool 2: Classifier Word Metric App
    st.markdown("### 📊 Classifier Word Metric App")
    st.markdown("**Analyze text classification performance with detailed word-level metrics**")
    st.markdown("Deep dive into your text data with comprehensive analysis at both statement and ID levels.")
    st.markdown("**Key Features:** Word-Level Analysis • Custom Classifiers • ID Aggregation • Export Configs")
    st.markdown("**Best for:** Content analysis, marketing research, sentiment studies")
    st.markdown("**Input:** CSV with ID and text columns | **Output:** Detailed metrics and statistics")
    st.markdown("")
    
    # Tool 3: Dictionary Classifier Creation App 
    st.markdown("### 📚 Dictionary Classifier Creation App")
    st.markdown("**Advanced text classification using customizable keyword dictionaries**")
    st.markdown("Classify text data with continuous variables, visualizations, and comprehensive reporting.")
    st.markdown("**Key Features:** Custom Dictionaries • Continuous Variables • Visual Analytics • Real-time Results")
    st.markdown("**Best for:** Content categorization, theme analysis, automated tagging")
    st.markdown("**Input:** CSV with text data | **Output:** Classifications with confidence scores")
    st.markdown("")
    
    # Tool 4: Dictionary Refinement App
    st.markdown("### 🏷️ Dictionary Refinement App")
    st.markdown("**Fast and straightforward text classification for quick insights**")
    st.markdown("Streamlined keyword matching with binary classifications and easy-to-understand results.")
    st.markdown("**Key Features:** Quick Setup • Binary Classification • Keyword Tracking • Simple Interface")
    st.markdown("**Best for:** Quick content screening, simple categorization, proof of concepts")
    st.markdown("**Input:** CSV with text column | **Output:** Binary classifications with matched keywords")
    st.markdown("")
    

    
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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📋 Data Preparation Tips**
        - **CSV Format**: Ensure your files have clear column headers
        - **Text Quality**: Clean text data performs better (remove excessive punctuation, fix encoding issues)
        - **File Size**: While we support large files, consider splitting very large datasets for better performance
        - **Column Names**: Use descriptive column names for easier identification
        """)
    
    with col2:
        st.markdown("""
        **🎯 Tool Selection Guide**
        - **Choose Join Table App** when you need to combine data from multiple sources
        - **Choose Classifier Word Metric Apps** for detailed text analysis with custom categories
        - **Choose Dictionary Classifier** for comprehensive text classification with visualizations
        - **Choose Simple Classifier** for quick, straightforward text categorization
        """)
    
    with col3:
        st.markdown("""
        **⚡ Performance Optimization**
        - **Large Files**: Process in smaller batches if you experience slowdowns
        - **Fuzzy Matching**: Adjust thresholds based on your data quality
        - **Custom Dictionaries**: Start with smaller keyword sets and expand based on results
        - **Browser Performance**: Close other tabs for better performance with large datasets
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("#### Ready to Transform Your Data? 🚀")
    st.markdown("Select any tool from the tabs above to get started with your analysis")
    st.markdown("*Built with ❤️ using Streamlit • No data is stored on our servers • Everything runs in your browser*")
    
    # Optional: Add current time/date
    current_time = datetime.now().strftime("%B %d, %Y • %I:%M %p")
    st.sidebar.markdown(f"**Current Session:** {current_time}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Quick Navigation:**
    - 🔗 Join Table App
    - 📊 Word Metrics  
    - 📚 Dictionary Classifier
    - 🏷️ Simple Classifier
    """)

if __name__ == "__main__":
    main()
