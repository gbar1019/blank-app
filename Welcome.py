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
    
    # Tool 1: CSV Table Joiner
    st.markdown("""
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
    </div>
    """, unsafe_allow_html=True)
    
    # Tool 2: Text Classification Word Metrics
    st.markdown("""
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
    </div>
    """, unsafe_allow_html=True)
    
    # Tool 3: Dictionary-Based Text Classifier
    st.markdown("""
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
    </div>
    """, unsafe_allow_html=True)
    
    # Tool 4: Simple Dictionary Classifier
    st.markdown("""
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
    st.sidebar.markdown("""
    **Quick Navigation:**
    - 🔗 Table Joiner
    - 📊 Word Metrics  
    - 📚 Dictionary Classifier
    - 🏷️ Simple Classifier
    """)

if __name__ == "__main__":
    main()
