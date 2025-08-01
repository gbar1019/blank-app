import streamlit as st
import pandas as pd
import re
import json
from typing import Dict, List, Union
from collections import defaultdict
import io
import plotly.express as px
import plotly.graph_objects as go

class DictionaryClassifier:
    """Enhanced dictionary-based text classifier with continuous variables."""
    
    def __init__(self, dictionaries: Dict[str, List[str]] = None):
        """Initialize classifier with dictionaries."""
        if dictionaries is None:
            # Default dictionaries
            self.dictionaries = {
                'personalized_products': ['custom', 'customized', 'personalized', 'bespoke', 'tailored', 'made to order'],
                'urgency_marketing': ['limited', 'limited time', 'limited run', 'limited edition', 'order now',
                                     'last chance', 'hurry', 'while supplies last', 'before they\'re gone',
                                     'selling out', 'selling fast', 'act now', 'don\'t wait', 'today only',
                                     'expires soon', 'final hours', 'almost gone'],
                'exclusive_marketing': ['exclusive', 'exclusively', 'exclusive offer', 'exclusive deal',
                                      'members only', 'vip', 'special access', 'invitation only',
                                      'premium', 'privileged', 'limited access', 'select customers',
                                      'insider', 'private sale', 'early access']
            }
        else:
            self.dictionaries = dictionaries
    
    def analyze_text(self, text: str) -> Dict[str, Union[int, float, List[str]]]:
        """Analyze single text with all categories."""
        if pd.isna(text):
            text = ""
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        total_words = len(words)
        
        results = {}
        
        for category, keywords in self.dictionaries.items():
            match_count = 0
            matched_keywords = []
            
            for keyword in keywords:
                # Use word boundaries for exact matching
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                matches = re.findall(pattern, text_lower)
                if matches:
                    match_count += len(matches)
                    matched_keywords.append(keyword)
            
            # Calculate percentage
            percentage = (match_count / total_words * 100) if total_words > 0 else 0
            
            # Store results
            results[f'{category}_word_count'] = match_count
            results[f'{category}_word_percentage'] = round(percentage, 4)
            results[f'{category}_binary'] = 1 if match_count > 0 else 0
            results[f'{category}_matched_keywords'] = matched_keywords
        
        results['_total_words'] = total_words
        return results
    
    def classify_dataframe(self, df: pd.DataFrame, text_column: str = 'Statement') -> pd.DataFrame:
        """Classify entire dataframe."""
        result_df = df.copy()
        
        # Apply analysis to each row
        analysis_results = df[text_column].apply(self.analyze_text)
        
        # Extract results into separate columns
        for category in self.dictionaries.keys():
            result_df[f'{category}_word_count'] = analysis_results.apply(lambda x: x[f'{category}_word_count'])
            result_df[f'{category}_word_percentage'] = analysis_results.apply(lambda x: x[f'{category}_word_percentage'])
            result_df[f'{category}_binary'] = analysis_results.apply(lambda x: x[f'{category}_binary'])
            result_df[f'{category}_matched_keywords'] = analysis_results.apply(lambda x: x[f'{category}_matched_keywords'])
        
        result_df['_total_words'] = analysis_results.apply(lambda x: x['_total_words'])
        
        return result_df
    
    def generate_summary(self, df: pd.DataFrame) -> Dict:
        """Generate analysis summary."""
        summary = {
            'total_posts': len(df),
            'category_frequency': {},
            'keyword_frequency': defaultdict(int),
            'avg_percentages': {}
        }
        
        for category in self.dictionaries.keys():
            # Count posts with matches
            posts_with_matches = (df[f'{category}_binary'] == 1).sum()
            percentage = (posts_with_matches / len(df) * 100) if len(df) > 0 else 0
            
            summary['category_frequency'][category] = {
                'posts': int(posts_with_matches),
                'percentage': round(percentage, 1)
            }
            
            # Average word percentage
            avg_percentage = df[f'{category}_word_percentage'].mean()
            summary['avg_percentages'][category] = round(avg_percentage, 4)
            
            # Count keyword frequencies
            for _, row in df.iterrows():
                matched_keywords = row[f'{category}_matched_keywords']
                if isinstance(matched_keywords, list):
                    for keyword in matched_keywords:
                        summary['keyword_frequency'][keyword] += 1
        
        return summary

def create_sample_data():
    """Create sample data for demonstration."""
    return pd.DataFrame([
        {'id': 'CKPXcrzsYTh', 'Statement': 'No problem.', 'likes': 0, 'comments': 0},
        {'id': 'B_ngLlvhNKJ', 'Statement': 'Refined in its look, soft, stretchy to the touch, and remarkably comfortable.', 'likes': 1, 'comments': 1},
        {'id': 'CNAS39lLnDd', 'Statement': 'But most importantly, I will continuously remember and practice the core values of excellence.', 'likes': 0, 'comments': 0},
        {'id': 'CMkjP5wLJlp', 'Statement': 'The whites and blues you\'ve been wearing for years are getting old and worn out.', 'likes': 0, 'comments': 0},
        {'id': 'CM8gKlLJcV', 'Statement': 'Congrats to the newlyweds, Jake & Micah!', 'likes': 0, 'comments': 0},
        {'id': 'TEST001', 'Statement': 'Our custom tailored suits are personalized just for you with bespoke fitting.', 'likes': 15, 'comments': 8},
        {'id': 'TEST002', 'Statement': 'Limited time only - exclusive offer for VIP members while supplies last!', 'likes': 23, 'comments': 12},
        {'id': 'TEST003', 'Statement': 'Classic timeless luxury style that never goes out of fashion.', 'likes': 7, 'comments': 3},
        {'id': 'TEST004', 'Statement': 'Get exclusive access to our premium personalized service today only!', 'likes': 31, 'comments': 15},
        {'id': 'TEST005', 'Statement': 'Made to order custom solutions for your individual needs.', 'likes': 12, 'comments': 5}
    ])

def main():
    st.set_page_config(
        page_title="Dictionary Text Classifier",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Dictionary-Based Text Classifier")
    st.markdown("**Analyze text data using customizable keyword dictionaries with continuous variables**")
    
    # Initialize session state
    if 'classifier' not in st.session_state:
        st.session_state.classifier = DictionaryClassifier()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
    
    # Sidebar for dictionary management
    with st.sidebar:
        st.header("📖 Dictionary Management")
        
        # Sample data option
        if st.button("📋 Load Sample Data"):
            sample_df = create_sample_data()
            st.session_state.sample_data = sample_df
            st.session_state.data_loaded = True
            st.session_state.analysis_completed = False
            st.success("Sample data loaded!")
        
        st.divider()
        
        # Dictionary management
        st.subheader("🏷️ Keyword Dictionaries")
        
        current_dictionaries = st.session_state.classifier.dictionaries
        
        # Add new dictionary
        with st.expander("➕ Add New Dictionary"):
            new_dict_name = st.text_input("Dictionary name:", key="new_dict")
            new_keywords = st.text_area("Keywords (one per line):", key="new_dict_keywords", height=100)
            
            if st.button("Add Dictionary") and new_dict_name and new_keywords:
                keywords_list = [k.strip() for k in new_keywords.split('\n') if k.strip()]
                st.session_state.classifier.dictionaries[new_dict_name] = keywords_list
                st.success(f"Added dictionary: {new_dict_name}")
                st.rerun()
        
        # Edit existing dictionaries
        st.write("**Current Dictionaries:**")
        for dict_name, keywords in current_dictionaries.items():
            with st.expander(f"📝 {dict_name.replace('_', ' ').title()} ({len(keywords)} keywords)"):
                # Show keywords
                current_keywords = '\n'.join(keywords)
                edited_keywords = st.text_area(
                    "Keywords:",
                    value=current_keywords,
                    key=f"edit_dict_{dict_name}",
                    height=150
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Update", key=f"update_{dict_name}"):
                        keywords_list = [k.strip() for k in edited_keywords.split('\n') if k.strip()]
                        st.session_state.classifier.dictionaries[dict_name] = keywords_list
                        st.success(f"Updated {dict_name}")
                        st.rerun()
                
                with col2:
                    if st.button("Remove", key=f"remove_{dict_name}"):
                        del st.session_state.classifier.dictionaries[dict_name]
                        st.success(f"Removed {dict_name}")
                        st.rerun()
        
        st.divider()
        
        # Import/Export dictionaries
        st.subheader("⚙️ Import/Export")
        
        # Export current dictionaries
        dict_json = json.dumps(st.session_state.classifier.dictionaries, indent=2)
        st.download_button(
            label="📥 Export Dictionaries (JSON)",
            data=dict_json,
            file_name="dictionaries.json",
            mime="application/json"
        )
        
        # Import dictionaries
        uploaded_dict = st.file_uploader("Upload dictionaries JSON:", type=['json'])
        if uploaded_dict is not None:
            try:
                dict_data = json.load(uploaded_dict)
                if st.button("Import Dictionaries"):
                    st.session_state.classifier.dictionaries = dict_data
                    st.success("Dictionaries imported successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error importing dictionaries: {e}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Upload", "🔍 Analysis", "📊 Results", "📈 Visualizations"])
    
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
                    st.subheader("🎯 Select Text Column")
                    text_column = st.selectbox("Select text column:", options=df.columns, key="text_col_select")
                    
                    if st.button("Load Data"):
                        st.session_state.uploaded_data = df
                        st.session_state.text_column = text_column
                        st.session_state.data_loaded = True
                        st.session_state.analysis_completed = False
                        st.success("Data loaded successfully!")
                        st.rerun()
                    
                    # Data preview
                    st.subheader("👀 Data Preview")
                    st.dataframe(df.head(10), height=300)
                    
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        with col2:
            st.subheader("📚 Current Dictionaries")
            for dict_name, keywords in current_dictionaries.items():
                with st.expander(f"{dict_name.replace('_', ' ').title()}"):
                    st.write(f"**{len(keywords)} keywords:**")
                    st.write(", ".join(keywords[:10]))
                    if len(keywords) > 10:
                        st.write(f"... and {len(keywords) - 10} more")
    
    with tab2:
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please upload data or load sample data first.")
        else:
            st.subheader("🔍 Run Classification Analysis")
            
            # Show loaded data info
            data_source = "Sample Data" if hasattr(st.session_state, 'sample_data') else "Uploaded Data"
            current_data = getattr(st.session_state, 'sample_data', None) or getattr(st.session_state, 'uploaded_data', None)
            text_column = getattr(st.session_state, 'text_column', 'Statement')
            
            if current_data is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Source", data_source)
                with col2:
                    st.metric("Total Rows", len(current_data))
                with col3:
                    st.metric("Dictionaries", len(st.session_state.classifier.dictionaries))
                
                st.subheader("📝 Analysis Configuration")
                st.write(f"**Text Column:** {text_column}")
                st.write(f"**Active Dictionaries:** {', '.join(st.session_state.classifier.dictionaries.keys())}")
                
                # Show sample text for preview
                st.subheader("📖 Sample Text Preview")
                sample_texts = current_data[text_column].head(3).tolist()
                for i, text in enumerate(sample_texts, 1):
                    st.write(f"**Sample {i}:** {text}")
                
                # Run analysis button
                if st.button("🚀 Run Classification", type="primary"):
                    with st.spinner("Running dictionary classification..."):
                        try:
                            classified_df = st.session_state.classifier.classify_dataframe(current_data, text_column)
                            summary = st.session_state.classifier.generate_summary(classified_df)
                            
                            st.session_state.classified_data = classified_df
                            st.session_state.summary = summary
                            st.session_state.analysis_completed = True
                            
                            st.success("✅ Classification completed successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error during classification: {e}")
    
    with tab3:
        if not st.session_state.analysis_completed:
            st.warning("⚠️ Please run the classification analysis first.")
        else:
            st.subheader("📊 Classification Results")
            
            classified_df = st.session_state.classified_data
            summary = st.session_state.summary
            
            # Overall statistics
            st.subheader("📈 Overall Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Posts", summary['total_posts'])
            with col2:
                st.metric("Total Words", int(classified_df['_total_words'].sum()))
            with col3:
                st.metric("Avg Words/Post", f"{classified_df['_total_words'].mean():.1f}")
            with col4:
                total_matches = sum([data['posts'] for data in summary['category_frequency'].values()])
                st.metric("Total Matches", total_matches)
            
            # Category frequency
            st.subheader("🏷️ Category Performance")
            
            category_data = []
            for category, data in summary['category_frequency'].items():
                category_data.append({
                    'Category': category.replace('_', ' ').title(),
                    'Posts with Matches': data['posts'],
                    'Percentage of Posts': f"{data['percentage']}%",
                    'Avg Word Percentage': f"{summary['avg_percentages'][category]:.4f}%",
                    'Total Keywords': len(st.session_state.classifier.dictionaries[category])
                })
            
            category_df = pd.DataFrame(category_data)
            st.dataframe(category_df, hide_index=True)
            
            # Top keywords
            st.subheader("🔑 Top Keywords")
            top_keywords = sorted(summary['keyword_frequency'].items(), key=lambda x: x[1], reverse=True)[:15]
            
            if top_keywords:
                keyword_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency'])
                st.dataframe(keyword_df, hide_index=True)
            
            # Detailed results
            st.subheader("📋 Detailed Classification Results")
            
            # Select columns to display
            basic_cols = [col for col in classified_df.columns if not any(suffix in col for suffix in ['_word_count', '_word_percentage', '_binary', '_matched_keywords'])]
            analysis_cols = [col for col in classified_df.columns if any(suffix in col for suffix in ['_word_count', '_word_percentage', '_binary'])]
            
            col1, col2 = st.columns(2)
            with col1:
                show_basic = st.checkbox("Show original data", value=True)
            with col2:
                show_analysis = st.checkbox("Show analysis results", value=True)
            
            display_cols = []
            if show_basic:
                display_cols.extend(basic_cols[:3])  # Limit to first 3 basic columns
            if show_analysis:
                display_cols.extend(analysis_cols)
            
            if display_cols:
                n_rows = st.slider("Number of rows to display:", 5, min(50, len(classified_df)), 10)
                st.dataframe(classified_df[display_cols].head(n_rows), height=400)
            
            # Download options
            st.subheader("💾 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Full results
                csv_full = classified_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results",
                    data=csv_full,
                    file_name="dictionary_classification_full.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Continuous variables only
                continuous_cols = [col for col in classified_df.columns 
                                 if '_word_count' in col or '_word_percentage' in col or '_binary' in col]
                if 'id' in classified_df.columns:
                    continuous_cols = ['id'] + continuous_cols
                continuous_cols.append('_total_words')
                
                continuous_df = classified_df[continuous_cols]
                csv_continuous = continuous_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Continuous Variables",
                    data=csv_continuous,
                    file_name="dictionary_classification_continuous.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Summary
                csv_summary = category_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary",
                    data=csv_summary,
                    file_name="dictionary_classification_summary.csv",
                    mime="text/csv"
                )
    
    with tab4:
        if not st.session_state.analysis_completed:
            st.warning("⚠️ Please run the classification analysis first.")
        else:
            st.subheader("📈 Classification Visualizations")
            
            classified_df = st.session_state.classified_data
            summary = st.session_state.summary
            
            # Category frequency bar chart
            st.subheader("📊 Category Frequency")
            
            categories = []
            posts = []
            percentages = []
            
            for category, data in summary['category_frequency'].items():
                categories.append(category.replace('_', ' ').title())
                posts.append(data['posts'])
                percentages.append(data['percentage'])
            
            fig_bar = px.bar(
                x=categories,
                y=posts,
                title="Posts with Matches by Category",
                labels={'x': 'Category', 'y': 'Number of Posts'},
                text=posts
            )
            fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Percentage pie chart
            st.subheader("🥧 Category Distribution")
            
            fig_pie = px.pie(
                values=posts,
                names=categories,
                title="Distribution of Matches Across Categories"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Word percentage heatmap
            st.subheader("🌡️ Average Word Percentages")
            
            avg_percentages = [summary['avg_percentages'][cat] for cat in summary['avg_percentages'].keys()]
            category_names = [cat.replace('_', ' ').title() for cat in summary['avg_percentages'].keys()]
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=[avg_percentages],
                x=category_names,
                y=['Average %'],
                colorscale='Blues',
                text=[[f"{val:.4f}%" for val in avg_percentages]],
                texttemplate="%{text}",
                textfont={"size": 12}
            ))
            
            fig_heatmap.update_layout(
                title="Average Word Percentages by Category",
                xaxis_title="Category",
                height=200
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Top keywords bar chart
            if summary['keyword_frequency']:
                st.subheader("🔑 Top Keywords Frequency")
                
                top_keywords = sorted(summary['keyword_frequency'].items(), key=lambda x: x[1], reverse=True)[:10]
                keywords, frequencies = zip(*top_keywords)
                
                fig_keywords = px.bar(
                    x=list(frequencies),
                    y=list(keywords),
                    orientation='h',
                    title="Top 10 Most Frequent Keywords",
                    labels={'x': 'Frequency', 'y': 'Keywords'},
                    text=frequencies
                )
                fig_keywords.update_traces(texttemplate='%{text}', textposition='outside')
                fig_keywords.update_layout(height=400)
                st.plotly_chart(fig_keywords, use_container_width=True)

if __name__ == "__main__":
    main()
