import streamlit as st
import pandas as pd
import re
import nltk
from typing import List, Dict, Optional
import time
from io import StringIO

# Download required NLTK data on first run
@st.cache_data
def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

class SentenceTokenizer:
    """Advanced sentence tokenizer with multiple methods and customization options."""
    
    def __init__(self):
        """Initialize the tokenizer."""
        download_nltk_data()
    
    def simple_tokenize(self, text: str, custom_patterns: List[str] = None) -> List[str]:
        """
        Simple regex-based tokenization.
        
        Args:
            text: Input text to tokenize
            custom_patterns: Optional custom split patterns
        
        Returns:
            List of tokenized sentences
        """
        if pd.isna(text):
            return []
        
        text = str(text).strip()
        if not text:
            return []
        
        # Extract hashtags first
        hashtags = re.findall(r'#\w+', text)
        clean_text = re.sub(r'#\w+', '', text)
        
        # Replace HTML breaks with sentence markers
        clean_text = re.sub(r'<br\s*/?>', ' [BREAK] ', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'<p\s*/?>', ' [PARA] ', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'</p>', ' [/PARA] ', clean_text, flags=re.IGNORECASE)
        
        # Default split patterns
        default_patterns = [
            r'(?<=[.!?])\s+',  # After sentence endings
            r'[,;:]',          # Commas, semicolons, colons
            r'\[BREAK\]',      # HTML breaks
            r'\[PARA\]',       # Paragraph tags
            r'\[/PARA\]'       # Closing paragraph tags
        ]
        
        # Add custom patterns if provided
        if custom_patterns:
            patterns = default_patterns + custom_patterns
        else:
            patterns = default_patterns
        
        # Combine patterns
        combined_pattern = '|'.join(patterns)
        
        # Split text
        parts = re.split(combined_pattern, clean_text)
        
        # Clean and filter parts
        sentences = []
        for part in parts:
            part = part.strip()
            # Skip empty parts and punctuation-only parts
            if part and not re.fullmatch(r'[.?!,:;"\'\-\s]+', part):
                sentences.append(part)
        
        # Add hashtags as separate sentence if they exist
        if hashtags:
            sentences.append(' '.join(hashtags))
        
        return sentences
    
    def nltk_tokenize(self, text: str) -> List[str]:
        """
        NLTK-based sentence tokenization.
        
        Args:
            text: Input text to tokenize
        
        Returns:
            List of tokenized sentences
        """
        if pd.isna(text):
            return []
        
        text = str(text).strip()
        if not text:
            return []
        
        # Clean HTML tags but preserve content
        clean_text = re.sub(r'<br\s*/?>', '. ', text, flags=re.IGNORECASE)
        clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
        
        try:
            sentences = nltk.sent_tokenize(clean_text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception:
            # Fallback to simple tokenization
            return self.simple_tokenize(text)
    
    def hybrid_tokenize(self, text: str, min_length: int = 5, max_length: int = 500) -> List[str]:
        """
        Hybrid tokenization combining NLTK and custom rules.
        
        Args:
            text: Input text to tokenize
            min_length: Minimum sentence length
            max_length: Maximum sentence length
        
        Returns:
            List of tokenized sentences
        """
        if pd.isna(text):
            return []
        
        # Start with NLTK tokenization
        nltk_sentences = self.nltk_tokenize(text)
        
        # Apply additional processing
        processed_sentences = []
        for sentence in nltk_sentences:
            # Split long sentences at commas if they exceed max_length
            if len(sentence) > max_length:
                parts = sentence.split(',')
                current_part = ""
                for part in parts:
                    if len(current_part + part) > max_length and current_part:
                        processed_sentences.append(current_part.strip())
                        current_part = part
                    else:
                        current_part += ("," if current_part else "") + part
                if current_part:
                    processed_sentences.append(current_part.strip())
            else:
                processed_sentences.append(sentence)
        
        # Filter by length
        filtered_sentences = [s for s in processed_sentences if min_length <= len(s) <= max_length]
        
        return filtered_sentences

def create_sample_data():
    """Create sample data for demonstration."""
    return pd.DataFrame([
        {
            'conversation_id': 'CONV001',
            'speaker': 'Customer',
            'message': 'Hello there! I need help with my order. It hasn\'t arrived yet, and I\'m getting worried. Can you please check the status? Thank you so much for your help.',
            'timestamp': '2023-01-01 10:00:00'
        },
        {
            'conversation_id': 'CONV001',
            'speaker': 'Agent',
            'message': 'Hi! I\'d be happy to help you with that. Let me check your order status right away. Could you please provide me with your order number? I\'ll look it up in our system.',
            'timestamp': '2023-01-01 10:01:00'
        },
        {
            'conversation_id': 'CONV002',
            'speaker': 'Customer',
            'message': 'Great product! Love the quality. Fast shipping too. <br>Would definitely recommend to others. Five stars! #awesome #quality #fast',
            'timestamp': '2023-01-01 11:00:00'
        },
        {
            'conversation_id': 'CONV002',
            'speaker': 'Agent',
            'message': 'Thank you so much for the positive feedback! We really appreciate customers like you. Your review means a lot to our team.',
            'timestamp': '2023-01-01 11:01:00'
        }
    ])

def main():
    st.set_page_config(
        page_title="Advanced Sentence Tokenizer",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 Advanced Sentence Tokenizer")
    st.markdown("**Transform your text data into individual sentences with advanced tokenization options**")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'tokenization_complete' not in st.session_state:
        st.session_state.tokenization_complete = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Tokenization Settings")
        
        # Sample data option
        if st.button("📋 Load Sample Data"):
            st.session_state.sample_data = create_sample_data()
            st.session_state.data_loaded = True
            st.session_state.tokenization_complete = False
            st.success("Sample data loaded!")
            st.rerun()
        
        st.divider()
        
        # Tokenization method
        st.subheader("🔧 Method Selection")
        tokenization_method = st.selectbox(
            "Choose tokenization method:",
            options=["Hybrid (Recommended)", "NLTK-based", "Simple Regex"],
            help="Different methods for splitting text into sentences"
        )
        
        # Advanced options
        st.subheader("🎛️ Advanced Options")
        
        if tokenization_method == "Hybrid (Recommended)":
            min_sentence_length = st.slider("Minimum sentence length:", 1, 50, 5)
            max_sentence_length = st.slider("Maximum sentence length:", 100, 1000, 500)
        
        # Custom patterns for simple tokenization
        if tokenization_method == "Simple Regex":
            with st.expander("Custom Split Patterns"):
                custom_patterns_text = st.text_area(
                    "Additional regex patterns (one per line):",
                    help="Add custom regex patterns to split sentences",
                    placeholder="\\n\\n\n\\t"
                )
                custom_patterns = [p.strip() for p in custom_patterns_text.split('\n') if p.strip()]
        else:
            custom_patterns = []
        
        # Output options
        st.subheader("📤 Output Options")
        include_original_text = st.checkbox("Include original text", value=True)
        include_sentence_stats = st.checkbox("Include sentence statistics", value=True)
        remove_empty_sentences = st.checkbox("Remove empty sentences", value=True)
    
    # Help section
    with st.expander("ℹ️ How to Use This App", expanded=False):
        st.markdown("""
        ### 📋 Step-by-Step Guide:
        
        1. **Upload CSV file** or use sample data
        2. **Select columns**:
           - **ID Column**: Identifies each original text/conversation
           - **Text Column**: Contains the text to be tokenized
           - **Speaker Column**: (Optional) Identifies the speaker
        3. **Configure settings** in the sidebar:
           - Choose tokenization method
           - Set sentence length limits
           - Add custom split patterns
        4. **Filter speakers** (if applicable)
        5. **Run tokenization** and download results
        
        ### 🔧 Tokenization Methods:
        
        - **Hybrid**: Combines NLTK with custom rules (best quality)
        - **NLTK-based**: Uses natural language processing
        - **Simple Regex**: Fast regex-based splitting
        
        ### 🎯 Use Cases:
        - **Data Preparation**: Create training data for ML models
        - **Text Analysis**: Break down conversations into analyzable units
        - **Content Processing**: Split articles or documents into sentences
        - **Research**: Analyze sentence-level patterns in text data
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📁 Data Input", "🔧 Processing", "📊 Results"])
    
    with tab1:
        st.subheader("📁 Upload Your Dataset")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    with st.spinner("Reading CSV file..."):
                        df = pd.read_csv(uploaded_file)
                    
                    st.session_state.uploaded_data = df
                    st.session_state.data_loaded = True
                    st.session_state.tokenization_complete = False
                    st.success(f"✅ File uploaded: {len(df)} rows, {len(df.columns)} columns")
                    
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        with col2:
            st.subheader("📊 File Requirements")
            st.info("""
            **Required columns:**
            - ID column (conversation/document ID)
            - Text column (content to tokenize)
            
            **Optional columns:**
            - Speaker column
            - Timestamp columns
            - Any other metadata
            """)
        
        # Show data preview
        if st.session_state.data_loaded:
            current_data = getattr(st.session_state, 'sample_data', None) or getattr(st.session_state, 'uploaded_data', None)
            
            st.subheader("👀 Data Preview")
            st.dataframe(current_data.head(10), height=300)
            
            # Data statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(current_data))
            with col2:
                st.metric("Columns", len(current_data.columns))
            with col3:
                # Estimate total text length
                text_cols = [col for col in current_data.columns if current_data[col].dtype == 'object']
                if text_cols:
                    total_chars = sum(current_data[text_cols[0]].astype(str).str.len())
                    st.metric("Est. Characters", f"{total_chars:,}")
    
    with tab2:
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please upload data or load sample data first.")
        else:
            st.subheader("🔧 Configure Tokenization")
            
            current_data = getattr(st.session_state, 'sample_data', None) or getattr(st.session_state, 'uploaded_data', None)
            
            # Column selection
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Required Columns:**")
                id_col = st.selectbox("Select ID column:", options=current_data.columns.tolist())
                text_col = st.selectbox("Select text column:", options=current_data.columns.tolist())
            
            with col2:
                st.write("**Optional Settings:**")
                speaker_col = st.selectbox(
                    "Select speaker column (optional):",
                    options=[None] + current_data.columns.tolist()
                )
            
            # Speaker filtering
            selected_speakers = []
            if speaker_col:
                unique_speakers = current_data[speaker_col].dropna().unique().tolist()
                selected_speakers = st.multiselect(
                    "🎯 Choose speakers to include:",
                    options=unique_speakers,
                    default=unique_speakers,
                    help="Only process text from selected speakers"
                )
            
            # Preview sample tokenization
            st.subheader("🔍 Tokenization Preview")
            
            if len(current_data) > 0:
                # Get a sample text
                sample_text = current_data[text_col].iloc[0]
                st.write("**Sample Input:**")
                st.write(f"*{sample_text}*")
                
                # Show tokenization result
                tokenizer = SentenceTokenizer()
                
                if tokenization_method == "Hybrid (Recommended)":
                    sample_sentences = tokenizer.hybrid_tokenize(sample_text, min_sentence_length, max_sentence_length)
                elif tokenization_method == "NLTK-based":
                    sample_sentences = tokenizer.nltk_tokenize(sample_text)
                else:  # Simple Regex
                    sample_sentences = tokenizer.simple_tokenize(sample_text, custom_patterns)
                
                st.write("**Tokenized Output:**")
                for i, sentence in enumerate(sample_sentences, 1):
                    st.write(f"{i}. *{sentence}*")
                
                st.info(f"This sample would produce {len(sample_sentences)} sentences.")
            
            # Processing button
            if st.button("🚀 Start Tokenization", type="primary", use_container_width=True):
                if not id_col or not text_col:
                    st.error("Please select both ID and text columns.")
                else:
                    with st.spinner("Tokenizing sentences..."):
                        try:
                            # Initialize tokenizer
                            tokenizer = SentenceTokenizer()
                            
                            # Progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Process data
                            data = []
                            total_rows = len(current_data)
                            
                            for idx, (_, row) in enumerate(current_data.iterrows()):
                                # Update progress
                                progress = (idx + 1) / total_rows
                                progress_bar.progress(progress)
                                status_text.text(f"Processing row {idx + 1}/{total_rows}")
                                
                                # Filter by speaker if needed
                                if speaker_col and selected_speakers and row[speaker_col] not in selected_speakers:
                                    continue
                                
                                # Tokenize based on selected method
                                if tokenization_method == "Hybrid (Recommended)":
                                    sentences = tokenizer.hybrid_tokenize(
                                        row[text_col], 
                                        min_sentence_length, 
                                        max_sentence_length
                                    )
                                elif tokenization_method == "NLTK-based":
                                    sentences = tokenizer.nltk_tokenize(row[text_col])
                                else:  # Simple Regex
                                    sentences = tokenizer.simple_tokenize(row[text_col], custom_patterns)
                                
                                # Filter empty sentences if requested
                                if remove_empty_sentences:
                                    sentences = [s for s in sentences if s.strip()]
                                
                                # Create entries
                                for i, sentence in enumerate(sentences, 1):
                                    entry = {
                                        'ID': row[id_col],
                                        'Sentence_ID': i,
                                        'Statement': sentence.strip()
                                    }
                                    
                                    # Add original text if requested
                                    if include_original_text:
                                        entry['Original_Text'] = row[text_col]
                                    
                                    # Add speaker if available
                                    if speaker_col:
                                        entry['Speaker'] = row[speaker_col]
                                    
                                    # Add sentence statistics if requested
                                    if include_sentence_stats:
                                        entry['Sentence_Length'] = len(sentence)
                                        entry['Word_Count'] = len(sentence.split())
                                    
                                    # Add other original columns
                                    for col in current_data.columns:
                                        if col not in [id_col, text_col, speaker_col] and col not in entry:
                                            entry[col] = row[col]
                                    
                                    data.append(entry)
                            
                            # Create result dataframe
                            result_df = pd.DataFrame(data)
                            st.session_state.tokenized_data = result_df
                            st.session_state.tokenization_complete = True
                            
                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.success("✅ Tokenization completed successfully!")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Error during tokenization: {e}")
                            st.exception(e)
    
    with tab3:
        if not st.session_state.tokenization_complete:
            st.warning("⚠️ Please complete tokenization first from the Processing tab.")
        else:
            st.subheader("📊 Tokenization Results")
            
            result_df = st.session_state.tokenized_data
            
            # Results summary
            st.subheader("📈 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Sentences", len(result_df))
            with col2:
                unique_ids = result_df['ID'].nunique()
                st.metric("Original Texts", unique_ids)
            with col3:
                avg_sentences = len(result_df) / unique_ids if unique_ids > 0 else 0
                st.metric("Avg Sentences/Text", f"{avg_sentences:.1f}")
            with col4:
                if 'Word_Count' in result_df.columns:
                    avg_words = result_df['Word_Count'].mean()
                    st.metric("Avg Words/Sentence", f"{avg_words:.1f}")
                else:
                    st.metric("Columns Generated", len(result_df.columns))
            
            # Sentence length distribution
            if 'Sentence_Length' in result_df.columns:
                st.subheader("📊 Sentence Length Distribution")
                
                # Create bins for sentence lengths
                length_bins = pd.cut(result_df['Sentence_Length'], bins=10)
                length_dist = length_bins.value_counts().sort_index()
                
                st.bar_chart(length_dist)
            
            # Results preview
            st.subheader("👀 Results Preview")
            
            # Column selection for display
            all_cols = list(result_df.columns)
            default_cols = ['ID', 'Sentence_ID', 'Statement']
            if 'Speaker' in all_cols:
                default_cols.append('Speaker')
            
            selected_cols = st.multiselect(
                "Select columns to display:",
                options=all_cols,
                default=[col for col in default_cols if col in all_cols],
                help="Choose which columns to show in the preview"
            )
            
            if selected_cols:
                n_rows = st.slider("Number of rows to display:", 5, min(50, len(result_df)), 10)
                st.dataframe(result_df[selected_cols].head(n_rows), height=400)
            
            # Sample comparison
            st.subheader("🔍 Before & After Example")
            if len(result_df) > 0:
                # Show original vs tokenized for first ID
                first_id = result_df['ID'].iloc[0]
                id_sentences = result_df[result_df['ID'] == first_id]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Text:**")
                    if 'Original_Text' in result_df.columns:
                        original = id_sentences['Original_Text'].iloc[0]
                        st.write(f"*{original}*")
                    else:
                        st.write("Original text not included in output")
                
                with col2:
                    st.write("**Tokenized Sentences:**")
                    for _, row in id_sentences.head(5).iterrows():
                        st.write(f"{row['Sentence_ID']}. *{row['Statement']}*")
                    
                    if len(id_sentences) > 5:
                        st.write(f"... and {len(id_sentences) - 5} more sentences")
            
            # Download section
            st.subheader("💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Full results
                csv_full = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv_full,
                    file_name='sentence_tokenized_full.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            
            with col2:
                # Essential columns only
                essential_cols = ['ID', 'Sentence_ID', 'Statement']
                if 'Speaker' in result_df.columns:
                    essential_cols.append('Speaker')
                
                csv_essential = result_df[essential_cols].to_csv(index=False)
                st.download_button(
                    label="📥 Download Essential Columns (CSV)",
                    data=csv_essential,
                    file_name='sentence_tokenized_essential.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            
            # Processing summary
            st.subheader("⚙️ Processing Summary")
            
            summary_data = {
                'Setting': [
                    'Tokenization Method',
                    'ID Column',
                    'Text Column',
                    'Speaker Column',
                    'Total Input Rows',
                    'Total Output Sentences',
                    'Include Original Text',
                    'Include Statistics'
                ],
                'Value': [
                    tokenization_method,
                    getattr(st.session_state, 'id_col', 'N/A'),
                    getattr(st.session_state, 'text_col', 'N/A'),
                    getattr(st.session_state, 'speaker_col', 'None'),
                    len(getattr(st.session_state, 'sample_data', getattr(st.session_state, 'uploaded_data', pd.DataFrame()))),
                    len(result_df),
                    'Yes' if include_original_text else 'No',
                    'Yes' if include_sentence_stats else 'No'
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, hide_index=True)

if __name__ == "__main__":
    main()
