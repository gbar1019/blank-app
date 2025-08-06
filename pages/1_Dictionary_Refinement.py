import streamlit as st
import pandas as pd
import re
import json
from typing import Dict, List, Set, Tuple
import io
from collections import defaultdict, Counter
import numpy as np

# Try to import plotly, fall back to matplotlib if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class DictionaryTester:
    """Test and refine keyword dictionaries for text classification."""
    
    def __init__(self):
        """Initialize with default dictionaries."""
        self.dictionaries = {
            'urgency_marketing': {
                'limited', 'limited time', 'limited run', 'limited edition', 'order now',
                'last chance', 'hurry', 'while supplies last', 'before they\'re gone',
                'selling out', 'selling fast', 'act now', 'don\'t wait', 'today only',
                'expires soon', 'final hours', 'almost gone', 'ending soon'
            },
            'exclusive_marketing': {
                'exclusive', 'exclusively', 'exclusive offer', 'exclusive deal',
                'members only', 'vip', 'special access', 'invitation only',
                'premium', 'privileged', 'limited access', 'select customers',
                'insider', 'private sale', 'early access'
            },
            'personalized_service': {
                'custom', 'customized', 'personalized', 'bespoke', 'tailored',
                'made to order', 'individual', 'personal', 'one-of-a-kind',
                'specially made', 'just for you', 'your style'
            },
            'discount_pricing': {
                'sale', 'discount', 'off', 'save', 'deal', 'special price',
                'reduced', 'clearance', 'markdown', 'bargain', 'cheap',
                'affordable', 'budget', 'value'
            },
            'social_proof': {
                'reviews', 'testimonials', 'customers love', 'rated', 'recommended',
                'trusted', 'popular', 'bestseller', 'award-winning', 'featured',
                'customer favorite', 'highly rated'
            }
        }
        self.test_results = {}
        self.df = None
        self.text_column = None
    
    def load_data(self, df: pd.DataFrame, text_column: str, ground_truth_column: str = None):
        """Load test data for dictionary evaluation."""
        self.df = df.copy()
        self.text_column = text_column
        self.ground_truth_column = ground_truth_column
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'text_column': text_column,
            'ground_truth_column': ground_truth_column
        }
    
    def classify_text(self, text: str, category: str) -> Dict:
        """Classify a single text for a specific category."""
        if pd.isna(text):
            text = ""
        
        text_lower = text.lower()
        keywords = self.dictionaries.get(category, set())
        
        matched_keywords = []
        total_matches = 0
        
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                matched_keywords.append(keyword)
                total_matches += len(matches)
        
        words = re.findall(r'\b\w+\b', text_lower)
        total_words = len(words)
        
        return {
            'matched_keywords': matched_keywords,
            'match_count': total_matches,
            'word_count': total_words,
            'match_percentage': (total_matches / total_words * 100) if total_words > 0 else 0,
            'binary_classification': 1 if total_matches > 0 else 0
        }
    
    def test_category(self, category: str) -> Dict:
        """Test a single category against the loaded data."""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        results = []
        keyword_performance = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'appearances': 0})
        
        for _, row in self.df.iterrows():
            text = str(row[self.text_column]) if pd.notna(row[self.text_column]) else ""
            classification = self.classify_text(text, category)
            
            # Add ground truth if available
            if self.ground_truth_column and self.ground_truth_column in self.df.columns:
                ground_truth = row[self.ground_truth_column]
                classification['ground_truth'] = ground_truth
                
                # Calculate confusion matrix for overall performance
                predicted = classification['binary_classification']
                if ground_truth == 1 and predicted == 1:
                    classification['result'] = 'tp'
                elif ground_truth == 0 and predicted == 1:
                    classification['result'] = 'fp'
                elif ground_truth == 1 and predicted == 0:
                    classification['result'] = 'fn'
                else:  # ground_truth == 0 and predicted == 0
                    classification['result'] = 'tn'
                
                # Track keyword-level performance
                for keyword in classification['matched_keywords']:
                    keyword_performance[keyword]['appearances'] += 1
                    if ground_truth == 1:
                        keyword_performance[keyword]['tp'] += 1
                    else:
                        keyword_performance[keyword]['fp'] += 1
                
                # Track keywords that should have appeared but didn't
                if ground_truth == 1 and classification['binary_classification'] == 0:
                    for keyword in self.dictionaries[category]:
                        if keyword not in classification['matched_keywords']:
                            keyword_performance[keyword]['fn'] += 1
            
            results.append(classification)
        
        # Calculate overall metrics
        if self.ground_truth_column and self.ground_truth_column in self.df.columns:
            tp = sum(1 for r in results if r.get('result') == 'tp')
            fp = sum(1 for r in results if r.get('result') == 'fp')
            fn = sum(1 for r in results if r.get('result') == 'fn')
            tn = sum(1 for r in results if r.get('result') == 'tn')
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
            
            overall_metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
            }
        else:
            overall_metrics = {}
        
        # Calculate keyword-level metrics
        keyword_metrics = {}
        for keyword, perf in keyword_performance.items():
            if perf['appearances'] > 0:
                keyword_precision = perf['tp'] / (perf['tp'] + perf['fp']) if (perf['tp'] + perf['fp']) > 0 else 0
                keyword_recall = perf['tp'] / (perf['tp'] + perf['fn']) if (perf['tp'] + perf['fn']) > 0 else 0
                keyword_f1 = 2 * (keyword_precision * keyword_recall) / (keyword_precision + keyword_recall) if (keyword_precision + keyword_recall) > 0 else 0
                
                keyword_metrics[keyword] = {
                    'precision': keyword_precision,
                    'recall': keyword_recall,
                    'f1_score': keyword_f1,
                    'appearances': perf['appearances'],
                    'tp': perf['tp'],
                    'fp': perf['fp'],
                    'fn': perf['fn']
                }
        
        return {
            'category': category,
            'overall_metrics': overall_metrics,
            'keyword_metrics': keyword_metrics,
            'classifications': results,
            'total_texts': len(results),
            'positive_classifications': sum(1 for r in results if r['binary_classification'] == 1)
        }
    
    def test_all_categories(self) -> Dict:
        """Test all categories against the loaded data."""
        all_results = {}
        for category in self.dictionaries.keys():
            all_results[category] = self.test_category(category)
        
        self.test_results = all_results
        return all_results
    
    def get_keyword_recommendations(self, category: str, min_appearances: int = 3) -> Dict:
        """Get recommendations for keyword optimization."""
        if category not in self.test_results:
            return {}
        
        results = self.test_results[category]
        keyword_metrics = results['keyword_metrics']
        
        recommendations = {
            'keep': [],
            'review': [],
            'remove': [],
            'missing': []
        }
        
        for keyword, metrics in keyword_metrics.items():
            if metrics['appearances'] >= min_appearances:
                if metrics['f1_score'] >= 0.7:
                    recommendations['keep'].append({
                        'keyword': keyword,
                        'f1_score': metrics['f1_score'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'appearances': metrics['appearances']
                    })
                elif metrics['f1_score'] >= 0.4:
                    recommendations['review'].append({
                        'keyword': keyword,
                        'f1_score': metrics['f1_score'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'appearances': metrics['appearances']
                    })
                else:
                    recommendations['remove'].append({
                        'keyword': keyword,
                        'f1_score': metrics['f1_score'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'appearances': metrics['appearances']
                    })
        
        # Sort recommendations by F1 score
        for rec_type in ['keep', 'review', 'remove']:
            recommendations[rec_type].sort(key=lambda x: x['f1_score'], reverse=True)
        
        return recommendations

def create_performance_charts(test_results: Dict) -> Dict:
    """Create performance visualization charts."""
    if not PLOTLY_AVAILABLE:
        return {}
    
    charts = {}
    
    # Overall Performance Comparison
    categories = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for category, results in test_results.items():
        if 'overall_metrics' in results and results['overall_metrics']:
            categories.append(category.replace('_', ' ').title())
            precisions.append(results['overall_metrics']['precision'])
            recalls.append(results['overall_metrics']['recall'])
            f1_scores.append(results['overall_metrics']['f1_score'])
    
    if categories:
        # Performance comparison bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Precision', x=categories, y=precisions))
        fig.add_trace(go.Bar(name='Recall', x=categories, y=recalls))
        fig.add_trace(go.Bar(name='F1 Score', x=categories, y=f1_scores))
        
        fig.update_layout(
            title='Category Performance Comparison',
            xaxis_title='Categories',
            yaxis_title='Score',
            barmode='group',
            height=400
        )
        charts['performance_comparison'] = fig
        
        # F1 Score ranking
        category_f1_data = list(zip(categories, f1_scores))
        category_f1_data.sort(key=lambda x: x[1], reverse=True)
        
        sorted_categories, sorted_f1s = zip(*category_f1_data)
        
        fig_ranking = px.bar(
            x=sorted_f1s,
            y=sorted_categories,
            orientation='h',
            title='Categories Ranked by F1 Score',
            labels={'x': 'F1 Score', 'y': 'Category'}
        )
        fig_ranking.update_layout(height=400)
        charts['f1_ranking'] = fig_ranking
    
    return charts

def main():
    st.set_page_config(
        page_title="Dictionary Refinement & Testing",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Dictionary Refinement & Testing")
    st.markdown("**Test and optimize your keyword dictionaries for maximum classification performance**")
    
    # Initialize session state
    if 'tester' not in st.session_state:
        st.session_state.tester = DictionaryTester()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'testing_completed' not in st.session_state:
        st.session_state.testing_completed = False
    
    # Sidebar for dictionary management
    with st.sidebar:
        st.header("📚 Dictionary Management")
        
        # Current dictionaries overview
        st.subheader("Current Dictionaries")
        for category, keywords in st.session_state.tester.dictionaries.items():
            st.write(f"**{category.replace('_', ' ').title()}**: {len(keywords)} keywords")
        
        st.divider()
        
        # Dictionary import/export
        st.subheader("⚙️ Import/Export")
        
        # Export current dictionaries
        dict_export = {}
        for category, keywords in st.session_state.tester.dictionaries.items():
            dict_export[category] = list(keywords)
        
        dict_json = json.dumps(dict_export, indent=2)
        st.download_button(
            label="📥 Export Dictionaries (JSON)",
            data=dict_json,
            file_name="refined_dictionaries.json",
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
                    
                    st.session_state.tester.dictionaries = imported_dicts
                    st.success("Dictionaries imported successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error importing dictionaries: {e}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Data Upload", 
        "📝 Dictionary Editor", 
        "🧪 Testing", 
        "📊 Results", 
        "🎯 Recommendations"
    ])
    
    with tab1:
        st.subheader("📁 Upload Test Data")
        
        with st.expander("ℹ️ Data Requirements"):
            st.markdown("""
            **Required Format:**
            - CSV file with a text column containing your content
            - Optional: Ground truth column with binary labels (0/1) for validation
            
            **Recommended:**
            - Use output from the Sentence Tokenizer app for best results
            - Include at least 100+ examples for reliable testing
            - Ground truth labels enable performance metrics calculation
            """)
        
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
                    st.subheader("🎯 Column Selection")
                    col_col1, col_col2 = st.columns(2)
                    
                    with col_col1:
                        text_column = st.selectbox("Select text column:", df.columns)
                    
                    with col_col2:
                        ground_truth_options = ["None"] + list(df.columns)
                        ground_truth_column = st.selectbox("Select ground truth column (optional):", ground_truth_options)
                        if ground_truth_column == "None":
                            ground_truth_column = None
                    
                    if st.button("Load Data for Testing"):
                        load_info = st.session_state.tester.load_data(
                            df, text_column, ground_truth_column
                        )
                        st.session_state.data_loaded = True
                        st.session_state.testing_completed = False
                        st.success("Data loaded successfully!")
                        
                        # Show load summary
                        st.write("**Load Summary:**")
                        st.write(f"- Rows: {load_info['rows']}")
                        st.write(f"- Text Column: {load_info['text_column']}")
                        if load_info['ground_truth_column']:
                            st.write(f"- Ground Truth Column: {load_info['ground_truth_column']}")
                            # Show distribution of ground truth
                            gt_dist = df[load_info['ground_truth_column']].value_counts()
                            st.write(f"- Ground Truth Distribution: {dict(gt_dist)}")
                        
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        with col2:
            st.subheader("📋 Current Status")
            if st.session_state.data_loaded:
                st.success("✅ Data loaded and ready for testing")
                if hasattr(st.session_state.tester, 'df'):
                    st.metric("Rows", len(st.session_state.tester.df))
                    st.metric("Text Column", st.session_state.tester.text_column)
                    if st.session_state.tester.ground_truth_column:
                        st.metric("Ground Truth", "Available")
                    else:
                        st.info("No ground truth - limited metrics available")
            else:
                st.warning("⚠️ No data loaded")
    
    with tab2:
        st.subheader("📝 Dictionary Editor")
        
        if not st.session_state.data_loaded:
            st.info("💡 Load test data first to see keyword performance during editing")
        
        # Category selection for editing
        categories = list(st.session_state.tester.dictionaries.keys())
        selected_category = st.selectbox("Select category to edit:", categories)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Keyword editing
            current_keywords = st.session_state.tester.dictionaries[selected_category]
            keywords_text = '\n'.join(sorted(current_keywords))
            
            edited_keywords = st.text_area(
                f"Keywords for {selected_category.replace('_', ' ').title()}:",
                value=keywords_text,
                height=300,
                help="One keyword per line. Multi-word phrases are supported."
            )
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Update Keywords", type="primary"):
                    new_keywords = {k.strip() for k in edited_keywords.split('\n') if k.strip()}
                    st.session_state.tester.dictionaries[selected_category] = new_keywords
                    st.success(f"Updated {selected_category}!")
                    st.rerun()
            
            with col_btn2:
                if st.button("Reset to Default"):
                    # Reset to default keywords (you can customize these defaults)
                    default_dict = DictionaryTester()
                    if selected_category in default_dict.dictionaries:
                        st.session_state.tester.dictionaries[selected_category] = default_dict.dictionaries[selected_category]
                        st.success(f"Reset {selected_category} to defaults!")
                        st.rerun()
        
        with col2:
            # Quick stats for current category
            st.subheader("📊 Category Stats")
            st.metric("Total Keywords", len(current_keywords))
            
            # Show sample keywords
            st.write("**Sample Keywords:**")
            sample_keywords = list(current_keywords)[:5]
            for kw in sample_keywords:
                st.write(f"• {kw}")
            if len(current_keywords) > 5:
                st.write(f"... and {len(current_keywords) - 5} more")
            
            # Quick test on sample data if available
            if st.session_state.data_loaded and st.button(f"Quick Test {selected_category}"):
                with st.spinner("Testing category..."):
                    results = st.session_state.tester.test_category(selected_category)
                    
                    st.write("**Quick Results:**")
                    st.write(f"Positive Classifications: {results['positive_classifications']}/{results['total_texts']}")
                    
                    if results['overall_metrics']:
                        st.write(f"F1 Score: {results['overall_metrics']['f1_score']:.3f}")
                        st.write(f"Precision: {results['overall_metrics']['precision']:.3f}")
                        st.write(f"Recall: {results['overall_metrics']['recall']:.3f}")
        
        st.divider()
        
        # Add new category
        with st.expander("➕ Add New Category"):
            new_category_name = st.text_input("Category name:")
            new_category_keywords = st.text_area("Keywords (one per line):")
            
            if st.button("Add Category") and new_category_name and new_category_keywords:
                keywords_set = {k.strip() for k in new_category_keywords.split('\n') if k.strip()}
                st.session_state.tester.dictionaries[new_category_name] = keywords_set
                st.success(f"Added {new_category_name}!")
                st.rerun()
    
    with tab3:
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load test data first from the Data Upload tab.")
        else:
            st.subheader("🧪 Dictionary Testing")
            
            # Show current setup
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test Data Rows", len(st.session_state.tester.df))
            with col2:
                st.metric("Categories", len(st.session_state.tester.dictionaries))
            with col3:
                has_ground_truth = st.session_state.tester.ground_truth_column is not None
                st.metric("Ground Truth", "Available" if has_ground_truth else "Not Available")
            
            # Testing options
            st.subheader("🎯 Testing Configuration")
            
            test_col1, test_col2 = st.columns(2)
            
            with test_col1:
                # Category selection
                categories_to_test = st.multiselect(
                    "Select categories to test:",
                    options=list(st.session_state.tester.dictionaries.keys()),
                    default=list(st.session_state.tester.dictionaries.keys()),
                    help="Select which categories to include in testing"
                )
            
            with test_col2:
                # Testing parameters
                min_appearances = st.number_input(
                    "Minimum keyword appearances:",
                    min_value=1,
                    max_value=20,
                    value=3,
                    help="Minimum times a keyword must appear to be included in analysis"
                )
            
            # Sample preview
            st.subheader("📖 Sample Preview")
            if len(st.session_state.tester.df) > 0:
                sample_texts = st.session_state.tester.df[st.session_state.tester.text_column].head(3).tolist()
                for i, text in enumerate(sample_texts, 1):
                    st.write(f"**Sample {i}:** {text}")
            
            # Run testing
            if st.button("🚀 Run Dictionary Testing", type="primary", use_container_width=True):
                with st.spinner("Testing dictionaries..."):
                    try:
                        # Filter dictionaries to test
                        original_dicts = st.session_state.tester.dictionaries.copy()
                        st.session_state.tester.dictionaries = {
                            k: v for k, v in original_dicts.items() 
                            if k in categories_to_test
                        }
                        
                        # Run testing
                        test_results = st.session_state.tester.test_all_categories()
                        st.session_state.testing_completed = True
                        
                        # Restore original dictionaries
                        st.session_state.tester.dictionaries = original_dicts
                        
                        st.success("✅ Testing completed!")
                        st.balloons()
                        
                        # Show quick summary
                        st.subheader("📊 Quick Summary")
                        summary_data = []
                        for category, results in test_results.items():
                            summary_row = {
                                'Category': category.replace('_', ' ').title(),
                                'Positive Classifications': f"{results['positive_classifications']}/{results['total_texts']}"
                            }
                            
                            if results['overall_metrics']:
                                summary_row.update({
                                    'F1 Score': f"{results['overall_metrics']['f1_score']:.3f}",
                                    'Precision': f"{results['overall_metrics']['precision']:.3f}",
                                    'Recall': f"{results['overall_metrics']['recall']:.3f}"
                                })
                            
                            summary_data.append(summary_row)
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, hide_index=True, use_container_width=True)
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error during testing: {e}")
    
    with tab4:
        if not st.session_state.testing_completed:
            st.warning("⚠️ Please run dictionary testing first from the Testing tab.")
        else:
            st.subheader("📊 Detailed Test Results")
            
            test_results = st.session_state.tester.test_results
            
            # Overall performance summary
            st.subheader("📈 Overall Performance Summary")
            
            summary_metrics = []
            for category, results in test_results.items():
                if results['overall_metrics']:
                    summary_metrics.append({
                        'Category': category.replace('_', ' ').title(),
                        'F1 Score': results['overall_metrics']['f1_score'],
                        'Precision': results['overall_metrics']['precision'],
                        'Recall': results['overall_metrics']['recall'],
                        'Accuracy': results['overall_metrics']['accuracy'],
                        'Positive Classifications': results['positive_classifications'],
                        'Total Texts': results['total_texts']
                    })
            
            if summary_metrics:
                summary_df = pd.DataFrame(summary_metrics)
                summary_df = summary_df.round(3)
                st.dataframe(summary_df, hide_index=True, use_container_width=True)
                
                # Performance charts
                if PLOTLY_AVAILABLE:
                    charts = create_performance_charts(test_results)
                    if charts:
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            if 'performance_comparison' in charts:
                                st.plotly_chart(charts['performance_comparison'], use_container_width=True)
                        
                        with chart_col2:
                            if 'f1_ranking' in charts:
                                st.plotly_chart(charts['f1_ranking'], use_container_width=True)
            
            # Detailed category results
            st.subheader("🔍 Detailed Category Analysis")
            
            selected_category_detail = st.selectbox(
                "Select category for detailed analysis:",
                options=list(test_results.keys()),
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            if selected_category_detail in test_results:
                category_results = test_results[selected_category_detail]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Overall Metrics:**")
                    if category_results['overall_metrics']:
                        metrics = category_results['overall_metrics']
                        metric_col1, metric_col2 = st.columns(2)
                        
                        with metric_col1:
                            st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
                            st.metric("Precision", f"{metrics['precision']:.3f}")
                        
                        with metric_col2:
                            st.metric("Recall", f"{metrics['recall']:.3f}")
                            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        
                        # Confusion matrix
                        st.write("**Confusion Matrix:**")
                        conf_matrix = pd.DataFrame({
                            'Predicted Positive': [metrics['tp'], metrics['fp']],
                            'Predicted Negative': [metrics['fn'], metrics['tn']]
                        }, index=['Actual Positive', 'Actual Negative'])
                        st.dataframe(conf_matrix)
                
                with col2:
                    st.write("**Classification Distribution:**")
                    total = category_results['total_texts']
                    positive = category_results['positive_classifications']
                    negative = total - positive
                    
                    st.metric("Total Texts", total)
                    st.metric("Positive Classifications", positive)
                    st.metric("Negative Classifications", negative)
                    st.metric("Positive Rate", f"{positive/total:.1%}")
                
                # Keyword performance table
                if category_results['keyword_metrics']:
                    st.subheader("🔑 Keyword Performance")
                    
                    keyword_data = []
                    for keyword, metrics in category_results['keyword_metrics'].items():
                        keyword_data.append({
                            'Keyword': keyword,
                            'F1 Score': metrics['f1_score'],
                            'Precision': metrics['precision'],
                            'Recall': metrics['recall'],
                            'Appearances': metrics['appearances'],
                            'True Positives': metrics['tp'],
                            'False Positives': metrics['fp']
                        })
                    
                    keyword_df = pd.DataFrame(keyword_data)
                    keyword_df = keyword_df.sort_values('F1 Score', ascending=False)
                    keyword_df = keyword_df.round(3)
                    
                    # Color coding for performance
                    def highlight_performance(val):
                        if isinstance(val, (int, float)):
                            if val >= 0.7:
                                return 'background-color: #d4edda'  # Light green
                            elif val >= 0.4:
                                return 'background-color: #fff3cd'  # Light yellow
                            else:
                                return 'background-color: #f8d7da'  # Light red
                        return ''
                    
                    styled_df = keyword_df.style.applymap(highlight_performance, subset=['F1 Score', 'Precision', 'Recall'])
                    st.dataframe(styled_df, hide_index=True, use_container_width=True)
                    
                    # Performance legend
                    st.write("**Performance Legend:**")
                    legend_col1, legend_col2, legend_col3 = st.columns(3)
                    with legend_col1:
                        st.success("🟢 Excellent (≥0.7)")
                    with legend_col2:
                        st.warning("🟡 Review (0.4-0.7)")
                    with legend_col3:
                        st.error("🔴 Poor (<0.4)")
            
            # Download detailed results
            st.subheader("💾 Download Results")
            
            # Prepare comprehensive results for download
            detailed_results = []
            for category, results in test_results.items():
                for classification in results['classifications']:
                    row = {
                        'category': category,
                        'text': st.session_state.tester.df.iloc[detailed_results.__len__() % len(st.session_state.tester.df)][st.session_state.tester.text_column] if len(detailed_results) < len(st.session_state.tester.df) else "",
                        'matched_keywords': ', '.join(classification['matched_keywords']),
                        'match_count': classification['match_count'],
                        'match_percentage': classification['match_percentage'],
                        'binary_classification': classification['binary_classification']
                    }
                    
                    if 'ground_truth' in classification:
                        row['ground_truth'] = classification['ground_truth']
                        row['result'] = classification.get('result', '')
                    
                    detailed_results.append(row)
            
            if detailed_results:
                results_df = pd.DataFrame(detailed_results)
                csv_results = results_df.to_csv(index=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 Download Detailed Results (CSV)",
                        data=csv_results,
                        file_name="dictionary_test_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Summary results
                    if summary_metrics:
                        summary_csv = pd.DataFrame(summary_metrics).to_csv(index=False)
                        st.download_button(
                            label="📥 Download Summary (CSV)",
                            data=summary_csv,
                            file_name="dictionary_performance_summary.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
    
    with tab5:
        if not st.session_state.testing_completed:
            st.warning("⚠️ Please run dictionary testing first to see recommendations.")
        else:
            st.subheader("🎯 Optimization Recommendations")
            
            test_results = st.session_state.tester.test_results
            
            # Category selection for recommendations
            rec_category = st.selectbox(
                "Select category for recommendations:",
                options=list(test_results.keys()),
                format_func=lambda x: x.replace('_', ' ').title(),
                key="rec_category"
            )
            
            if rec_category in test_results:
                recommendations = st.session_state.tester.get_keyword_recommendations(rec_category)
                
                # Display recommendations in columns
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    # Keep and Review keywords
                    st.subheader("✅ Keep These Keywords")
                    if recommendations['keep']:
                        keep_data = []
                        for item in recommendations['keep']:
                            keep_data.append({
                                'Keyword': item['keyword'],
                                'F1 Score': f"{item['f1_score']:.3f}",
                                'Precision': f"{item['precision']:.3f}",
                                'Appearances': item['appearances']
                            })
                        
                        keep_df = pd.DataFrame(keep_data)
                        st.dataframe(keep_df, hide_index=True, use_container_width=True)
                    else:
                        st.info("No high-performing keywords found. Consider adding more specific terms.")
                    
                    st.subheader("⚠️ Review These Keywords")
                    if recommendations['review']:
                        review_data = []
                        for item in recommendations['review']:
                            review_data.append({
                                'Keyword': item['keyword'],
                                'F1 Score': f"{item['f1_score']:.3f}",
                                'Precision': f"{item['precision']:.3f}",
                                'Appearances': item['appearances']
                            })
                        
                        review_df = pd.DataFrame(review_data)
                        st.dataframe(review_df, hide_index=True, use_container_width=True)
                        st.info("💡 These keywords have moderate performance. Consider refining or replacing them.")
                    else:
                        st.success("No keywords need review!")
                
                with rec_col2:
                    # Remove keywords
                    st.subheader("❌ Consider Removing")
                    if recommendations['remove']:
                        remove_data = []
                        for item in recommendations['remove']:
                            remove_data.append({
                                'Keyword': item['keyword'],
                                'F1 Score': f"{item['f1_score']:.3f}",
                                'Precision': f"{item['precision']:.3f}",
                                'Appearances': item['appearances']
                            })
                        
                        remove_df = pd.DataFrame(remove_data)
                        st.dataframe(remove_df, hide_index=True, use_container_width=True)
                        st.warning("⚠️ These keywords are causing more false positives than correct classifications.")
                    else:
                        st.success("No keywords recommended for removal!")
                    
                    # Overall category health
                    st.subheader("📊 Category Health")
                    
                    total_keywords = len(st.session_state.tester.dictionaries[rec_category])
                    keep_count = len(recommendations['keep'])
                    review_count = len(recommendations['review'])
                    remove_count = len(recommendations['remove'])
                    
                    health_metrics = [
                        {"Metric": "Total Keywords", "Value": total_keywords},
                        {"Metric": "High Performing", "Value": keep_count},
                        {"Metric": "Need Review", "Value": review_count},
                        {"Metric": "Poor Performing", "Value": remove_count},
                        {"Metric": "Health Score", "Value": f"{(keep_count / max(total_keywords, 1)):.1%}"}
                    ]
                    
                    health_df = pd.DataFrame(health_metrics)
                    st.dataframe(health_df, hide_index=True, use_container_width=True)
                
                # Actionable recommendations
                st.subheader("💡 Actionable Next Steps")
                
                next_steps = []
                
                if recommendations['remove']:
                    next_steps.append(f"🔴 **Remove {len(recommendations['remove'])} poor-performing keywords** to reduce false positives")
                
                if recommendations['review']:
                    next_steps.append(f"🟡 **Review {len(recommendations['review'])} moderate-performing keywords** - consider more specific variations")
                
                if len(recommendations['keep']) < 5:
                    next_steps.append("🟢 **Add more high-quality keywords** - your dictionary might be too small")
                
                overall_performance = test_results[rec_category]['overall_metrics']
                if overall_performance and overall_performance['f1_score'] < 0.6:
                    next_steps.append("📈 **Overall F1 score is low** - consider fundamental keyword strategy revision")
                
                if overall_performance and overall_performance['precision'] < 0.7:
                    next_steps.append("🎯 **Precision is low** - focus on more specific, less ambiguous keywords")
                
                if overall_performance and overall_performance['recall'] < 0.7:
                    next_steps.append("🔍 **Recall is low** - add more synonym variations and related terms")
                
                if next_steps:
                    for step in next_steps:
                        st.markdown(step)
                else:
                    st.success("🎉 Your dictionary is performing well! Consider minor tweaks based on the keyword analysis above.")
                
                # Auto-optimization option
                st.divider()
                st.subheader("🤖 Auto-Optimization")
                
                st.warning("⚠️ **Experimental Feature**: Auto-optimization will automatically remove poor-performing keywords. Make sure to backup your dictionary first!")
                
                auto_col1, auto_col2 = st.columns(2)
                
                with auto_col1:
                    remove_threshold = st.slider(
                        "F1 Score threshold for removal:",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.3,
                        step=0.1,
                        help="Keywords below this F1 score will be removed"
                    )
                
                with auto_col2:
                    min_appearances_auto = st.number_input(
                        "Minimum appearances for removal:",
                        min_value=1,
                        value=3,
                        help="Only remove keywords that appear at least this many times"
                    )
                
                if st.button("🤖 Auto-Optimize Dictionary", type="secondary"):
                    # Find keywords to remove based on threshold
                    keywords_to_remove = []
                    for item in recommendations['remove']:
                        if item['f1_score'] < remove_threshold and item['appearances'] >= min_appearances_auto:
                            keywords_to_remove.append(item['keyword'])
                    
                    if keywords_to_remove:
                        # Remove keywords from dictionary
                        current_keywords = st.session_state.tester.dictionaries[rec_category]
                        optimized_keywords = current_keywords - set(keywords_to_remove)
                        st.session_state.tester.dictionaries[rec_category] = optimized_keywords
                        
                        st.success(f"✅ Removed {len(keywords_to_remove)} poor-performing keywords:")
                        for kw in keywords_to_remove:
                            st.write(f"• {kw}")
                        
                        st.info("💡 Re-run testing to see the impact of optimization!")
                        st.rerun()
                    else:
                        st.info("No keywords meet the removal criteria.")
            
            # Cross-category analysis
            st.divider()
            st.subheader("🔄 Cross-Category Analysis")
            
            # Find keywords that appear in multiple categories
            all_keywords = {}
            for category, keywords in st.session_state.tester.dictionaries.items():
                for keyword in keywords:
                    if keyword not in all_keywords:
                        all_keywords[keyword] = []
                    all_keywords[keyword].append(category)
            
            # Find overlapping keywords
            overlapping_keywords = {k: v for k, v in all_keywords.items() if len(v) > 1}
            
            if overlapping_keywords:
                st.write("**Keywords appearing in multiple categories:**")
                overlap_data = []
                for keyword, categories in overlapping_keywords.items():
                    overlap_data.append({
                        'Keyword': keyword,
                        'Categories': ', '.join([cat.replace('_', ' ').title() for cat in categories]),
                        'Count': len(categories)
                    })
                
                overlap_df = pd.DataFrame(overlap_data)
                overlap_df = overlap_df.sort_values('Count', ascending=False)
                st.dataframe(overlap_df, hide_index=True, use_container_width=True)
                
                st.warning("⚠️ Overlapping keywords may cause classification conflicts. Consider making them more category-specific.")
            else:
                st.success("✅ No keyword overlaps detected between categories!")
            
            # Export optimized dictionary
            st.divider()
            st.subheader("📥 Export Optimized Dictionary")
            
            optimized_dict = {}
            for category in st.session_state.tester.dictionaries:
                if category in test_results:
                    recommendations = st.session_state.tester.get_keyword_recommendations(category)
                    # Include only high-performing and review keywords
                    optimized_keywords = []
                    for item in recommendations['keep'] + recommendations['review']:
                        optimized_keywords.append(item['keyword'])
                    optimized_dict[category] = optimized_keywords
                else:
                    # Keep original if not tested
                    optimized_dict[category] = list(st.session_state.tester.dictionaries[category])
            
            optimized_json = json.dumps(optimized_dict, indent=2)
            st.download_button(
                label="📥 Download Optimized Dictionary",
                data=optimized_json,
                file_name="optimized_dictionary.json",
                mime="application/json",
                help="Dictionary with poor-performing keywords filtered out",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
