import streamlit as st
import pandas as pd
import re
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import io
import json

class ClassifierWordMetrics:
    """
    Analyze text classification performance with detailed word-level metrics.
    Provides both statement-level and ID-level aggregated analysis.
    """
    
    def __init__(self):
        """Initialize the classifier with default keyword mappings."""
        self.keyword_mappings = {
            'urgency_marketing': ['limited', 'hurry', 'now', 'today only', 'expires', 'final', 'last chance'],
            'exclusive_marketing': ['exclusive', 'vip', 'members only', 'private', 'special access', 'premium'],
            'personalized_service': ['custom', 'personalized', 'tailored', 'bespoke', 'individual', 'personal'],
            'discount_pricing': ['sale', 'discount', 'off', 'save', 'deal', 'special price', 'reduced'],
            'social_proof': ['reviews', 'testimonials', 'customers love', 'rated', 'recommended', 'trusted'],
            'local_business': ['local', 'community', 'neighborhood', 'nearby', 'hometown', 'area'],
            'gratitude_reflection': ['thank', 'grateful', 'appreciate', 'blessed', 'thankful', 'honored']
        }
        
        self.df = None
        self.id_column = None
        self.text_column = None
        self.results = {}
    
    def load_data(self, df: pd.DataFrame, id_column: str, text_column: str):
        """Load data and configure analysis columns."""
        self.df = df.copy()
        self.id_column = id_column
        self.text_column = text_column
        
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'id_column': id_column,
            'text_column': text_column,
            'classifiers': list(self.keyword_mappings.keys())
        }
    
    def set_keyword_mappings(self, mappings: Dict[str, List[str]]):
        """Set custom keyword mappings for classifiers."""
        self.keyword_mappings = mappings
    
    def add_classifier(self, name: str, keywords: List[str]):
        """Add a new classifier with keywords."""
        self.keyword_mappings[name] = keywords
    
    def remove_classifier(self, name: str):
        """Remove a classifier."""
        if name in self.keyword_mappings:
            del self.keyword_mappings[name]
    
    def calculate_statement_metrics(self) -> pd.DataFrame:
        """Calculate word-level metrics for each statement."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        statement_metrics = []
        
        for _, row in self.df.iterrows():
            text = str(row[self.text_column]).lower() if pd.notna(row[self.text_column]) else ""
            words = re.findall(r'\b\w+\b', text)
            total_words = len(words)
            
            metrics = {
                self.id_column: row[self.id_column],
                self.text_column: row[self.text_column],
                'total_words': total_words
            }
            
            # Calculate metrics for each classifier
            for classifier, keywords in self.keyword_mappings.items():
                word_count = 0
                for keyword in keywords:
                    # Use word boundaries for exact matching
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    matches = re.findall(pattern, text)
                    word_count += len(matches)
                
                metrics[f'{classifier}_word_count'] = word_count
                metrics[f'{classifier}_word_pct'] = (word_count / total_words) if total_words > 0 else 0
            
            statement_metrics.append(metrics)
        
        statement_df = pd.DataFrame(statement_metrics)
        self.results['statement_level'] = statement_df
        
        return statement_df
    
    def calculate_id_metrics(self) -> pd.DataFrame:
        """Calculate aggregated metrics at the ID level."""
        if 'statement_level' not in self.results:
            raise ValueError("Statement metrics not calculated. Call calculate_statement_metrics() first.")
        
        statement_df = self.results['statement_level']
        
        # Group by ID
        id_groups = statement_df.groupby(self.id_column)
        
        # Calculate total corpus words for percentage calculations
        total_corpus_words = statement_df['total_words'].sum()
        
        id_metrics = []
        
        for id_value, group in id_groups:
            metrics = {
                self.id_column: id_value,
                'total_statements': len(group),
                'total_words': group['total_words'].sum(),
                'avg_words_per_statement': group['total_words'].mean()
            }
            
            # Calculate aggregated metrics for each classifier
            for classifier in self.keyword_mappings.keys():
                word_count_col = f'{classifier}_word_count'
                
                # Count statements with this classifier
                statements_with_classifier = (group[word_count_col] > 0).sum()
                
                # Total words for this classifier across all statements for this ID
                total_classifier_words = group[word_count_col].sum()
                
                metrics[f'{classifier}_stmt_count'] = statements_with_classifier
                metrics[f'{classifier}_word_count'] = total_classifier_words
                metrics[f'{classifier}_pct_of_corpus'] = (total_classifier_words / total_corpus_words) if total_corpus_words > 0 else 0
                metrics[f'{classifier}_avg_pct_within_ID'] = (total_classifier_words / metrics['total_words']) if metrics['total_words'] > 0 else 0
            
            id_metrics.append(metrics)
        
        id_df = pd.DataFrame(id_metrics)
        self.results['id_level'] = id_df
        
        return id_df
    
    def analyze_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run complete analysis and return both statement and ID level results."""
        statement_df = self.calculate_statement_metrics()
        id_df = self.calculate_id_metrics()
        return statement_df, id_df
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for the analysis."""
        if 'statement_level' not in self.results or 'id_level' not in self.results:
            return {}
        
        statement_df = self.results['statement_level']
        id_df = self.results['id_level']
        
        # Overall Statistics
        total_statements = len(statement_df)
        total_ids = len(id_df)
        total_words = statement_df['total_words'].sum()
        avg_words_per_statement = statement_df['total_words'].mean()
        avg_statements_per_id = statement_df.groupby(self.id_column).size().mean()
        
        # Classifier Performance
        classifier_stats = []
        for classifier in self.keyword_mappings.keys():
            word_count_col = f'{classifier}_word_count'
            word_pct_col = f'{classifier}_word_pct'
            
            statements_with_classifier = (statement_df[word_count_col] > 0).sum()
            total_classifier_words = statement_df[word_count_col].sum()
            avg_pct_per_statement = statement_df[word_pct_col].mean() * 100
            
            classifier_stats.append({
                'classifier': classifier.replace('_', ' ').title(),
                'statements_with_classifier': statements_with_classifier,
                'total_words': total_classifier_words,
                'avg_pct_per_statement': avg_pct_per_statement,
                'keywords': ', '.join(self.keyword_mappings[classifier])
            })
        
        return {
            'overall': {
                'total_statements': total_statements,
                'total_ids': total_ids,
                'total_words': total_words,
                'avg_words_per_statement': avg_words_per_statement,
                'avg_statements_per_id': avg_statements_per_id
            },
            'classifiers': classifier_stats
        }

def main():
    st.set_page_config(
        page_title="Text Classification Word Metrics",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Text Classification Word Metrics Analyzer")
    st.markdown("**Analyze text classification performance with detailed word-level metrics**")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ClassifierWordMetrics()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Keyword mappings management
        st.subheader("🏷️ Keyword Mappings")
        
        # Show current classifiers
        current_mappings = st.session_state.analyzer.keyword_mappings
        
        # Classifier management
        with st.expander("📝 Edit Classifiers"):
            # Add new classifier
            st.write("**Add New Classifier:**")
            new_classifier_name = st.text_input("Classifier name:", key="new_classifier")
            new_keywords = st.text_area("Keywords (one per line):", key="new_keywords")
            
            if st.button("Add Classifier") and new_classifier_name and new_keywords:
                keywords_list = [k.strip() for k in new_keywords.split('\n') if k.strip()]
                st.session_state.analyzer.add_classifier(new_classifier_name, keywords_list)
                st.success(f"Added classifier: {new_classifier_name}")
                st.rerun()
            
            st.divider()
            
            # Edit existing classifiers
            st.write("**Edit Existing Classifiers:**")
            for classifier_name, keywords in current_mappings.items():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**{classifier_name.replace('_', ' ').title()}:**")
                        current_keywords = '\n'.join(keywords)
                        edited_keywords = st.text_area(
                            "Keywords:",
                            value=current_keywords,
                            key=f"edit_{classifier_name}",
                            height=100
                        )
                    
                    with col2:
                        st.write("")  # Spacer
                        if st.button("Update", key=f"update_{classifier_name}"):
                            keywords_list = [k.strip() for k in edited_keywords.split('\n') if k.strip()]
                            st.session_state.analyzer.keyword_mappings[classifier_name] = keywords_list
                            st.success(f"Updated {classifier_name}")
                            st.rerun()
                        
                        if st.button("Remove", key=f"remove_{classifier_name}"):
                            st.session_state.analyzer.remove_classifier(classifier_name)
                            st.success(f"Removed {classifier_name}")
                            st.rerun()
                    
                    st.divider()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📁 Data Upload", "📊 Analysis", "📈 Results"])
    
    with tab1:
        st.subheader("📁 Upload Your Dataset")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ File uploaded: {len(df)} rows, {len(df.columns)} columns")
                    
                    # Column selection
                    st.subheader("🎯 Select Columns")
                    col_col1, col_col2 = st.columns(2)
                    
                    with col_col1:
                        id_column = st.selectbox("Select ID column:", options=df.columns, key="id_col_select")
                    
                    with col_col2:
                        text_column = st.selectbox("Select text column:", options=df.columns, key="text_col_select")
                    
                    if st.button("Load Data"):
                        load_info = st.session_state.analyzer.load_data(df, id_column, text_column)
                        st.session_state.data_loaded = True
                        st.session_state.analysis_completed = False
                        st.session_state.uploaded_data = df
                        st.success("Data loaded successfully!")
                        st.rerun()
                    
                    # Data preview
                    st.subheader("👀 Data Preview")
                    st.dataframe(df.head(10), height=300)
                    
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        with col2:
            st.subheader("📋 Current Classifiers")
            for classifier_name, keywords in current_mappings.items():
                with st.expander(f"{classifier_name.replace('_', ' ').title()} ({len(keywords)} keywords)"):
                    st.write(", ".join(keywords))
    
    with tab2:
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please upload data first.")
        else:
            st.subheader("🔄 Run Analysis")
            
            # Show loaded data info
            current_data = getattr(st.session_state, 'uploaded_data', None)
            
            if current_data is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Source", "Uploaded Data")
                with col2:
                    st.metric("Total Rows", len(current_data))
                with col3:
                    st.metric("Classifiers", len(st.session_state.analyzer.keyword_mappings))
                
                st.subheader("📝 Preview Configuration")
                st.write(f"**ID Column:** {st.session_state.analyzer.id_column}")
                st.write(f"**Text Column:** {st.session_state.analyzer.text_column}")
                st.write(f"**Active Classifiers:** {', '.join(st.session_state.analyzer.keyword_mappings.keys())}")
                
                # Run analysis button
                if st.button("🚀 Run Analysis", type="primary"):
                    with st.spinner("Running word metrics analysis..."):
                        try:
                            statement_results, id_results = st.session_state.analyzer.analyze_all()
                            st.session_state.analysis_completed = True
                            st.success("✅ Analysis completed successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {e}")
    
    with tab3:
        if not st.session_state.analysis_completed:
            st.warning("⚠️ Please run the analysis first.")
        else:
            st.subheader("📈 Analysis Results")
            
            # Get summary statistics
            summary_stats = st.session_state.analyzer.get_summary_stats()
            
            if summary_stats:
                # Overall statistics
                st.subheader("📊 Overall Statistics")
                overall = summary_stats['overall']
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Statements", f"{overall['total_statements']:,}")
                with col2:
                    st.metric("Total IDs", f"{overall['total_ids']:,}")
                with col3:
                    st.metric("Total Words", f"{overall['total_words']:,}")
                with col4:
                    st.metric("Avg Words/Statement", f"{overall['avg_words_per_statement']:.1f}")
                with col5:
                    st.metric("Avg Statements/ID", f"{overall['avg_statements_per_id']:.1f}")
                
                # Classifier performance
                st.subheader("🎯 Classifier Performance")
                classifier_df = pd.DataFrame(summary_stats['classifiers'])
                classifier_df['avg_pct_per_statement'] = classifier_df['avg_pct_per_statement'].round(2)
                
                st.dataframe(
                    classifier_df[['classifier', 'statements_with_classifier', 'total_words', 'avg_pct_per_statement']],
                    column_config={
                        'classifier': 'Classifier',
                        'statements_with_classifier': 'Statements with Keywords',
                        'total_words': 'Total Keywords Found',
                        'avg_pct_per_statement': 'Avg % per Statement'
                    },
                    hide_index=True
                )
                
                # Detailed results tabs
                st.subheader("📋 Detailed Results")
                result_tab1, result_tab2 = st.tabs(["Statement Level", "ID Level"])
                
                with result_tab1:
                    st.write("**Statement-level metrics:**")
                    statement_df = st.session_state.analyzer.results['statement_level']
                    
                    # Show number of rows to display
                    n_rows = st.slider("Number of rows to display:", 5, min(50, len(statement_df)), 10, key="stmt_rows")
                    
                    # Select columns to display
                    all_cols = list(statement_df.columns)
                    default_cols = [st.session_state.analyzer.id_column, st.session_state.analyzer.text_column, 'total_words']
                    
                    # Add first few classifier columns
                    for classifier in list(st.session_state.analyzer.keyword_mappings.keys())[:2]:
                        default_cols.extend([f'{classifier}_word_count', f'{classifier}_word_pct'])
                    
                    selected_cols = st.multiselect(
                        "Select columns to display:",
                        options=all_cols,
                        default=[col for col in default_cols if col in all_cols],
                        key="stmt_cols"
                    )
                    
                    if selected_cols:
                        st.dataframe(statement_df[selected_cols].head(n_rows), height=400)
                
                with result_tab2:
                    st.write("**ID-level aggregated metrics:**")
                    id_df = st.session_state.analyzer.results['id_level']
                    
                    # Show number of rows to display
                    n_rows_id = st.slider("Number of rows to display:", 5, min(50, len(id_df)), 10, key="id_rows")
                    
                    # Select columns to display
                    all_id_cols = list(id_df.columns)
                    default_id_cols = [st.session_state.analyzer.id_column, 'total_statements', 'total_words', 'avg_words_per_statement']
                    
                    # Add first few classifier columns
                    for classifier in list(st.session_state.analyzer.keyword_mappings.keys())[:2]:
                        default_id_cols.extend([f'{classifier}_stmt_count', f'{classifier}_avg_pct_within_ID'])
                    
                    selected_id_cols = st.multiselect(
                        "Select columns to display:",
                        options=all_id_cols,
                        default=[col for col in default_id_cols if col in all_id_cols],
                        key="id_cols"
                    )
                    
                    if selected_id_cols:
                        st.dataframe(id_df[selected_id_cols].head(n_rows_id), height=400)
                
                # Download results
                st.subheader("💾 Download Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Statement level CSV
                    csv_statement = statement_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Statement-Level CSV",
                        data=csv_statement,
                        file_name="statement_level_metrics.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # ID level CSV
                    csv_id = id_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download ID-Level CSV",
                        data=csv_id,
                        file_name="id_level_metrics.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    # Summary CSV
                    csv_summary = classifier_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary CSV",
                        data=csv_summary,
                        file_name="analysis_summary.csv",
                        mime="text/csv"
                    )
                
                # Configuration export/import
                st.subheader("⚙️ Configuration Management")
                
                config_col1, config_col2 = st.columns(2)
                
                with config_col1:
                    st.write("**Export Configuration:**")
                    config_json = json.dumps(st.session_state.analyzer.keyword_mappings, indent=2)
                    st.download_button(
                        label="📥 Download Keyword Mappings (JSON)",
                        data=config_json,
                        file_name="keyword_mappings.json",
                        mime="application/json"
                    )
                
                with config_col2:
                    st.write("**Import Configuration:**")
                    uploaded_config = st.file_uploader("Upload keyword mappings JSON:", type=['json'])
                    
                    if uploaded_config is not None:
                        try:
                            config_data = json.load(uploaded_config)
                            if st.button("Import Configuration"):
                                st.session_state.analyzer.set_keyword_mappings(config_data)
                                st.success("Configuration imported successfully!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error importing configuration: {e}")

if __name__ == "__main__":
    main()
