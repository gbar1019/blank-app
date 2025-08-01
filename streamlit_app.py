import streamlit as st
import pandas as pd
import re
from typing import Dict, Set, List
import io
import json

class SimpleDictionaryClassifier:
    """Simple dictionary-based text classifier with keyword matching."""
    
    def __init__(self):
        """Initialize with default dictionaries."""
        self.dictionaries = {
            'urgency_marketing': {
                'limited', 'limited time', 'limited run', 'limited edition', 'order now',
                'last chance', 'hurry', 'while supplies last', 'before they\'re gone',
                'selling out', 'selling fast', 'act now', 'don\'t wait', 'today only',
                'expires soon', 'final hours', 'almost gone'
            },
            'exclusive_marketing': {
                'exclusive', 'exclusively', 'exclusive offer', 'exclusive deal',
                'members only', 'vip', 'special access', 'invitation only',
                'premium', 'privileged', 'limited access', 'select customers',
                'insider', 'private sale', 'early access'
            }
        }
    
    def classify_text(self, text: str) -> Dict[str, List[str]]:
        """
        Classify text using dictionary matching.
        
        Args:
            text: Input text to classify
        
        Returns:
            Dict of {category: list of matched keywords}
        """
        if pd.isna(text):
            return {category: [] for category in self.dictionaries.keys()}
        
        text_lower = text.lower()
        results = {}
        
        for category, keywords in self.dictionaries.items():
            matched = []
            for keyword in keywords:
                # Use word boundaries for exact matches
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    matched.append(keyword)
            results[category] = matched
        
        return results
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'Statement') -> pd.DataFrame:
        """
        Process dataframe and add classification columns.
        
        Args:
            df: Input dataframe
            text_column: Name of column containing text to classify
        
        Returns:
            Dataframe with added classification columns
        """
        df = df.copy()
        
        # Apply classification to each row
        classifications = df[text_column].apply(self.classify_text)
        
        # Create columns for each category
        for category in self.dictionaries.keys():
            # Matched keywords
            df[f'{category}_keywords'] = classifications.apply(lambda x: x[category])
            # Count of matches
            df[f'{category}_count'] = df[f'{category}_keywords'].apply(len)
            # Binary indicator
            df[f'{category}_present'] = df[f'{category}_count'] > 0
        
        return df
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for the classification results."""
        summary = {}
        
        for category in self.dictionaries.keys():
            count_col = f'{category}_count'
            present_col = f'{category}_present'
            
            total_matches = df[count_col].sum()
            rows_with_matches = df[present_col].sum()
            percentage = (rows_with_matches / len(df) * 100) if len(df) > 0 else 0
            
            summary[category] = {
                'total_matches': int(total_matches),
                'rows_with_matches': int(rows_with_matches),
                'percentage': round(percentage, 1),
                'keywords_count': len(self.dictionaries[category])
            }
        
        return summary

def create_sample_data():
    """Create sample data for demonstration."""
    return pd.DataFrame([
        {'ID': 'POST001', 'Statement': 'Limited time offer - get it before it\'s gone!', 'category': 'marketing'},
        {'ID': 'POST002', 'Statement': 'Exclusive deal for VIP members only today.', 'category': 'marketing'},
        {'ID': 'POST003', 'Statement': 'Great product with excellent quality and service.', 'category': 'general'},
        {'ID': 'POST004', 'Statement': 'Hurry up! Final hours to get this special access.', 'category': 'marketing'},
        {'ID': 'POST005', 'Statement': 'Thank you for your continued support and loyalty.', 'category': 'service'},
        {'ID': 'POST006', 'Statement': 'Order now while supplies last - limited edition available!', 'category': 'marketing'},
        {'ID': 'POST007', 'Statement': 'Premium quality service for our privileged customers.', 'category': 'service'},
        {'ID': 'POST008', 'Statement': 'Regular updates and news about our products and services.', 'category': 'general'},
        {'ID': 'POST009', 'Statement': 'Don\'t wait - this exclusive offer expires soon!', 'category': 'marketing'},
        {'ID': 'POST010', 'Statement': 'Join our community for early access to new features.', 'category': 'community'}
    ])

def main():
    st.set_page_config(
        page_title="Simple Dictionary Classifier",
        page_icon="🏷️",
        layout="wide"
    )
    
    st.title("🏷️ Simple Dictionary Text Classifier")
    st.markdown("**Fast and easy text classification using keyword dictionaries**")
    
    # Initialize session state
    if 'classifier' not in st.session_state:
        st.session_state.classifier = SimpleDictionaryClassifier()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
    
    # Sidebar for quick actions and dictionary view
    with st.sidebar:
        st.header("🚀 Quick Actions")
        
        # Load sample data
        if st.button("📋 Load Sample Data", type="primary"):
            sample_df = create_sample_data()
            st.session_state.sample_data = sample_df
            st.session_state.data_loaded = True
            st.session_state.analysis_completed = False
            st.success("Sample data loaded!")
            st.rerun()
        
        st.divider()
        
        # Dictionary overview
        st.header("📚 Current Dictionaries")
        
        for category, keywords in st.session_state.classifier.dictionaries.items():
            with st.expander(f"📖 {category.replace('_', ' ').title()} ({len(keywords)} keywords)"):
                keywords_text = '\n'.join(sorted(keywords))
                updated_keywords = st.text_area(
                    "Keywords (one per line):",
                    value=keywords_text,
                    height=200,
                    key=f"dict_{category}"
                )
                
                if st.button(f"Update {category.replace('_', ' ').title()}", key=f"update_{category}"):
                    new_keywords = {k.strip() for k in updated_keywords.split('\n') if k.strip()}
                    st.session_state.classifier.dictionaries[category] = new_keywords
                    st.success(f"Updated {category}!")
                    st.rerun()
        
        st.divider()
        
        # Add new dictionary
        with st.expander("➕ Add New Dictionary"):
            new_dict_name = st.text_input("Dictionary name:")
            new_dict_keywords = st.text_area("Keywords (one per line):")
            
            if st.button("Add Dictionary") and new_dict_name and new_dict_keywords:
                keywords_set = {k.strip() for k in new_dict_keywords.split('\n') if k.strip()}
                st.session_state.classifier.dictionaries[new_dict_name] = keywords_set
                st.success(f"Added {new_dict_name}!")
                st.rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📁 Data Input", "🔍 Classification", "📊 Results"])
    
    with tab1:
        st.subheader("📁 Upload Your Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ File uploaded: {len(df)} rows, {len(df.columns)} columns")
                    
                    # Show data preview
                    st.subheader("👀 Data Preview")
                    st.dataframe(df.head(), height=200)
                    
                    # Column selection
                    st.subheader("🎯 Select Text Column")
                    text_column = st.selectbox("Choose the column containing text to classify:", df.columns)
                    
                    if st.button("Load Data"):
                        st.session_state.uploaded_data = df
                        st.session_state.text_column = text_column
                        st.session_state.data_loaded = True
                        st.session_state.analysis_completed = False
                        st.success("Data loaded successfully!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        with col2:
            st.subheader("📝 Data Requirements")
            st.info("""
            **Your CSV should contain:**
            - At least one text column to classify
            - Any additional columns you want to keep
            
            **Supported formats:**
            - CSV files with headers
            - Text columns with any content
            """)
            
            if st.session_state.data_loaded:
                current_data = getattr(st.session_state, 'sample_data', None) or getattr(st.session_state, 'uploaded_data', None)
                if current_data is not None:
                    st.success(f"✅ Data loaded: {len(current_data)} rows")
    
    with tab2:
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first from the Data Input tab.")
        else:
            st.subheader("🔍 Run Classification")
            
            # Show current setup
            current_data = getattr(st.session_state, 'sample_data', None) or getattr(st.session_state, 'uploaded_data', None)
            text_column = getattr(st.session_state, 'text_column', 'Statement')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows to Process", len(current_data))
            with col2:
                st.metric("Text Column", text_column)
            with col3:
                st.metric("Dictionaries", len(st.session_state.classifier.dictionaries))
            
            # Show sample texts
            st.subheader("📖 Sample Texts")
            sample_texts = current_data[text_column].head(3).tolist()
            for i, text in enumerate(sample_texts, 1):
                st.write(f"**{i}.** {text}")
            
            # Classification button
            if st.button("🚀 Start Classification", type="primary", use_container_width=True):
                with st.spinner("Classifying texts..."):
                    try:
                        classified_df = st.session_state.classifier.process_dataframe(current_data, text_column)
                        summary_stats = st.session_state.classifier.get_summary_stats(classified_df)
                        
                        st.session_state.classified_data = classified_df
                        st.session_state.summary_stats = summary_stats
                        st.session_state.analysis_completed = True
                        
                        st.success("✅ Classification completed!")
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error during classification: {e}")
    
    with tab3:
        if not st.session_state.analysis_completed:
            st.warning("⚠️ Please run classification first from the Classification tab.")
        else:
            st.subheader("📊 Classification Results")
            
            classified_df = st.session_state.classified_data
            summary_stats = st.session_state.summary_stats
            
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            
            # Create metrics display
            cols = st.columns(len(summary_stats))
            for i, (category, stats) in enumerate(summary_stats.items()):
                with cols[i]:
                    st.metric(
                        label=category.replace('_', ' ').title(),
                        value=f"{stats['rows_with_matches']} rows",
                        delta=f"{stats['percentage']}% of data"
                    )
            
            # Detailed breakdown
            st.subheader("📋 Detailed Breakdown")
            
            breakdown_data = []
            for category, stats in summary_stats.items():
                breakdown_data.append({
                    'Category': category.replace('_', ' ').title(),
                    'Rows with Matches': stats['rows_with_matches'],
                    'Total Matches': stats['total_matches'],
                    'Percentage of Rows': f"{stats['percentage']}%",
                    'Keywords in Dictionary': stats['keywords_count']
                })
            
            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, hide_index=True, use_container_width=True)
            
            # Results preview
            st.subheader("👀 Results Preview")
            
            # Show column selection
            all_cols = list(classified_df.columns)
            original_cols = [col for col in all_cols if not any(suffix in col for suffix in ['_keywords', '_count', '_present'])]
            result_cols = [col for col in all_cols if any(suffix in col for suffix in ['_keywords', '_count', '_present'])]
            
            col1, col2 = st.columns(2)
            with col1:
                show_original = st.checkbox("Show original columns", value=True)
            with col2:
                show_results = st.checkbox("Show classification results", value=True)
            
            display_cols = []
            if show_original:
                display_cols.extend(original_cols[:3])  # Show first 3 original columns
            if show_results:
                display_cols.extend(result_cols)
            
            if display_cols:
                n_rows = st.slider("Number of rows to display:", 5, min(20, len(classified_df)), 10)
                st.dataframe(classified_df[display_cols].head(n_rows), height=300)
            
            # Sample matches
            st.subheader("🎯 Sample Matches")
            
            for category in st.session_state.classifier.dictionaries.keys():
                matches_df = classified_df[classified_df[f'{category}_present'] == True]
                if len(matches_df) > 0:
                    st.write(f"**{category.replace('_', ' ').title()} Examples:**")
                    
                    sample_matches = matches_df.head(3)
                    text_col = getattr(st.session_state, 'text_column', 'Statement')
                    
                    for _, row in sample_matches.iterrows():
                        keywords = ', '.join(row[f'{category}_keywords'])
                        st.write(f"• *{row[text_col]}* → **Keywords:** {keywords}")
                    
                    if len(matches_df) > 3:
                        st.write(f"... and {len(matches_df) - 3} more matches")
                    st.write("")
            
            # Download results
            st.subheader("💾 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Full results
                csv_full = classified_df.to_csv(index=False)
                st.download_button(
                    label="📥 Full Results (CSV)",
                    data=csv_full,
                    file_name="classified_results_full.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Classification only
                class_cols = [col for col in classified_df.columns if any(suffix in col for suffix in ['_count', '_present'])]
                if 'ID' in classified_df.columns:
                    class_cols = ['ID'] + class_cols
                
                class_df = classified_df[class_cols]
                csv_class = class_df.to_csv(index=False)
                st.download_button(
                    label="📥 Classification Only (CSV)",
                    data=csv_class,
                    file_name="classified_results_classification.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                # Summary
                csv_summary = breakdown_df.to_csv(index=False)
                st.download_button(
                    label="📥 Summary (CSV)",
                    data=csv_summary,
                    file_name="classification_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Export dictionaries
            st.subheader("⚙️ Export Configuration")
            
            # Convert sets to lists for JSON serialization
            dict_export = {}
            for category, keywords in st.session_state.classifier.dictionaries.items():
                dict_export[category] = list(keywords)
            
            dict_json = json.dumps(dict_export, indent=2)
            st.download_button(
                label="📥 Export Dictionaries (JSON)",
                data=dict_json,
                file_name="dictionaries.json",
                mime="application/json"
            )
            
            # Import dictionaries
            uploaded_dict = st.file_uploader("📤 Import Dictionaries (JSON):", type=['json'])
            if uploaded_dict is not None:
                try:
                    dict_data = json.load(uploaded_dict)
                    if st.button("Import Dictionaries"):
                        # Convert lists back to sets
                        imported_dicts = {}
                        for category, keywords in dict_data.items():
                            imported_dicts[category] = set(keywords)
                        
                        st.session_state.classifier.dictionaries = imported_dicts
                        st.success("Dictionaries imported successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error importing dictionaries: {e}")

if __name__ == "__main__":
    main()
