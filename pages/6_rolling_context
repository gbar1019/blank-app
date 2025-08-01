import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import time
from typing import List, Dict, Optional

def process_rolling_context(df: pd.DataFrame, id_col: str, text_col: str, 
                          window_size: int, speaker_col: Optional[str] = None, 
                          selected_speakers: Optional[List[str]] = None,
                          include_future: bool = False, future_window: int = 0,
                          context_separator: str = " ") -> pd.DataFrame:
    """
    Process rolling context with optimized performance and additional features.
    
    Args:
        df: Input dataframe
        id_col: Column name for conversation ID
        text_col: Column name for text/statements
        window_size: Number of previous rows to include as context
        speaker_col: Optional speaker column name
        selected_speakers: List of speakers to include
        include_future: Whether to include future context
        future_window: Number of future rows to include
        context_separator: Separator for joining context
    
    Returns:
        Processed dataframe with context columns
    """
    # Sort data by ID and reset index
    df_sorted = df.sort_values(by=[id_col]).reset_index(drop=True)
    
    # Initialize result list
    result_rows = []
    
    # Get unique conversation IDs
    unique_ids = df_sorted[id_col].unique()
    
    # Process each conversation
    for conv_id in unique_ids:
        # Filter conversation data
        conv_df = df_sorted[df_sorted[id_col] == conv_id].reset_index(drop=True)
        
        # Process each row in conversation
        for i in range(len(conv_df)):
            current_row = conv_df.loc[i]
            
            # Filter by speaker if specified
            if speaker_col and selected_speakers and current_row[speaker_col] not in selected_speakers:
                continue
            
            # Get past context
            past_start = max(0, i - window_size)
            past_window = conv_df.loc[past_start:i - 1] if i > 0 else pd.DataFrame()
            
            # Filter past context by speakers if specified
            if speaker_col and selected_speakers and not past_window.empty:
                past_window = past_window[past_window[speaker_col].isin(selected_speakers)]
            
            # Get future context if requested
            future_context = ""
            future_speaker_history = ""
            if include_future and future_window > 0:
                future_end = min(len(conv_df), i + future_window + 1)
                future_window_df = conv_df.loc[i + 1:future_end - 1] if i < len(conv_df) - 1 else pd.DataFrame()
                
                if speaker_col and selected_speakers and not future_window_df.empty:
                    future_window_df = future_window_df[future_window_df[speaker_col].isin(selected_speakers)]
                
                if not future_window_df.empty:
                    future_context = context_separator.join(future_window_df[text_col].astype(str).tolist())
                    if speaker_col:
                        future_speaker_history = " | ".join(future_window_df[speaker_col].astype(str).tolist())
            
            # Create context string
            past_context = context_separator.join(past_window[text_col].astype(str).tolist()) if not past_window.empty else ""
            
            # Create entry dictionary
            entry = {
                id_col: conv_id,
                'Row_Index': i,
                'Statement': current_row[text_col],
                'Past_Context': past_context,
                'Context_Length': len(past_window),
                'Has_Context': len(past_window) > 0
            }
            
            # Add speaker information if available
            if speaker_col:
                entry['Speaker'] = current_row[speaker_col]
                if not past_window.empty:
                    past_speaker_history = " | ".join(past_window[speaker_col].astype(str).tolist())
                    entry['Past_Speaker_History'] = past_speaker_history
                else:
                    entry['Past_Speaker_History'] = ""
            
            # Add future context if requested
            if include_future:
                entry['Future_Context'] = future_context
                entry['Future_Context_Length'] = len(future_window_df) if 'future_window_df' in locals() else 0
                if speaker_col:
                    entry['Future_Speaker_History'] = future_speaker_history
            
            # Add all original columns (except the ones we're processing)
            for col in df.columns:
                if col not in [id_col, text_col, speaker_col] and col not in entry:
                    entry[col] = current_row[col]
            
            result_rows.append(entry)
    
    return pd.DataFrame(result_rows)

def create_sample_data():
    """Create sample conversation data for demonstration."""
    return pd.DataFrame([
        {'conversation_id': 'CONV001', 'speaker': 'Customer', 'statement': 'Hello, I need help with my order.', 'timestamp': '2023-01-01 10:00:00'},
        {'conversation_id': 'CONV001', 'speaker': 'Agent', 'statement': 'Hi! I\'d be happy to help. What\'s your order number?', 'timestamp': '2023-01-01 10:01:00'},
        {'conversation_id': 'CONV001', 'speaker': 'Customer', 'statement': 'It\'s order #12345. I haven\'t received it yet.', 'timestamp': '2023-01-01 10:02:00'},
        {'conversation_id': 'CONV001', 'speaker': 'Agent', 'statement': 'Let me check that for you. I see it was shipped yesterday.', 'timestamp': '2023-01-01 10:03:00'},
        {'conversation_id': 'CONV001', 'speaker': 'Customer', 'statement': 'Great! When should I expect delivery?', 'timestamp': '2023-01-01 10:04:00'},
        {'conversation_id': 'CONV002', 'speaker': 'Customer', 'statement': 'I want to return this product.', 'timestamp': '2023-01-01 11:00:00'},
        {'conversation_id': 'CONV002', 'speaker': 'Agent', 'statement': 'I can help with that. What\'s the reason for the return?', 'timestamp': '2023-01-01 11:01:00'},
        {'conversation_id': 'CONV002', 'speaker': 'Customer', 'statement': 'The size doesn\'t fit properly.', 'timestamp': '2023-01-01 11:02:00'},
        {'conversation_id': 'CONV002', 'speaker': 'Agent', 'statement': 'No problem. I\'ll start the return process for you.', 'timestamp': '2023-01-01 11:03:00'},
    ])

def main():
    st.set_page_config(
        page_title="Rolling Context Generator",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Rolling Context Generator")
    st.markdown("**Create contextual datasets from conversational data with advanced features**")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Sample data option
        if st.button("📋 Load Sample Data"):
            st.session_state.sample_data = create_sample_data()
            st.session_state.data_loaded = True
            st.session_state.processing_complete = False
            st.success("Sample data loaded!")
            st.rerun()
        
        st.divider()
        
        # Advanced options
        st.subheader("🔧 Advanced Options")
        
        include_future = st.checkbox(
            "Include future context", 
            help="Include statements that come after the current statement"
        )
        
        future_window = 0
        if include_future:
            future_window = st.number_input(
                "Future window size:", 
                min_value=1, 
                value=2, 
                step=1,
                help="Number of future statements to include"
            )
        
        context_separator = st.selectbox(
            "Context separator:",
            options=[" ", " | ", " || ", "\\n"],
            help="How to join multiple context statements"
        )
        
        # Replace \\n with actual newline
        if context_separator == "\\n":
            context_separator = "\n"
    
    # Help section
    with st.expander("ℹ️ How to Use This App", expanded=False):
        st.markdown("""
        ### 📋 Step-by-Step Guide:
        
        1. **Upload your CSV file** or use sample data
           - File should contain conversational data (one statement per row)
           - Must have columns for: ID, text/statements, and optionally speaker
        
        2. **Configure your settings**:
           - **ID Column**: Groups conversations together
           - **Text Column**: Contains the statements/sentences
           - **Speaker Column**: Optional - identifies who said what
           - **Window Size**: How many previous statements to include as context
        
        3. **Advanced Features**:
           - **Future Context**: Include statements that come after current one
           - **Speaker Filtering**: Only include specific speakers in context
           - **Custom Separators**: Choose how context statements are joined
        
        4. **Generate and Download**: Process data and save results as CSV
        
        ### 🎯 Use Cases:
        - **Customer Service**: Analyze support conversations with context
        - **Chat Analysis**: Understand conversation flow and patterns
        - **Training Data**: Create context-aware datasets for ML models
        - **Conversation Mining**: Extract insights from multi-turn dialogues
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📁 Data Input", "⚙️ Processing", "📊 Results"])
    
    with tab1:
        st.subheader("📁 Upload Your Dataset")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.uploaded_data = df
                    st.session_state.data_loaded = True
                    st.session_state.processing_complete = False
                    st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")
                    
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        with col2:
            st.subheader("📊 Data Requirements")
            st.info("""
            **Required columns:**
            - ID column (conversation groups)
            - Text column (statements/sentences)
            
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
            
            # Data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(current_data))
            with col2:
                st.metric("Columns", len(current_data.columns))
            with col3:
                if len(current_data.columns) > 0:
                    # Try to identify potential ID column
                    potential_id_cols = [col for col in current_data.columns if 'id' in col.lower()]
                    if potential_id_cols:
                        unique_ids = current_data[potential_id_cols[0]].nunique()
                        st.metric("Unique Groups", unique_ids)
                    else:
                        st.metric("Columns Available", len(current_data.columns))
    
    with tab2:
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please upload data or load sample data first.")
        else:
            st.subheader("⚙️ Configure Processing Settings")
            
            current_data = getattr(st.session_state, 'sample_data', None) or getattr(st.session_state, 'uploaded_data', None)
            
            # Column selection
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Required Columns:**")
                id_col = st.selectbox("Select ID Column:", options=current_data.columns.tolist())
                text_col = st.selectbox("Select Text Column:", options=current_data.columns.tolist())
            
            with col2:
                st.write("**Optional Settings:**")
                speaker_col = st.selectbox(
                    "Select Speaker Column (Optional):", 
                    options=["(None)"] + current_data.columns.tolist()
                )
                
                window_size = st.number_input(
                    "Past Context Window Size:", 
                    min_value=0, 
                    value=3, 
                    step=1,
                    help="Number of previous statements to include"
                )
            
            # Speaker filtering
            selected_speakers = []
            if speaker_col != "(None)":
                unique_speakers = current_data[speaker_col].dropna().unique().tolist()
                selected_speakers = st.multiselect(
                    "Select Speaker(s) to include in context:", 
                    options=unique_speakers, 
                    default=unique_speakers,
                    help="Only include these speakers when building context"
                )
            
            # Preview configuration
            st.subheader("📋 Configuration Preview")
            
            config_col1, config_col2, config_col3 = st.columns(3)
            with config_col1:
                st.info(f"**ID Column:** {id_col}")
                st.info(f"**Text Column:** {text_col}")
            with config_col2:
                st.info(f"**Window Size:** {window_size}")
                st.info(f"**Future Context:** {'Yes' if include_future else 'No'}")
            with config_col3:
                speaker_info = f"{speaker_col}" if speaker_col != "(None)" else "None"
                st.info(f"**Speaker Column:** {speaker_info}")
                if selected_speakers:
                    st.info(f"**Filtered Speakers:** {len(selected_speakers)}")
            
            # Process button
            if st.button("🚀 Generate Rolling Context", type="primary", use_container_width=True):
                if not id_col or not text_col:
                    st.error("Please select both ID and Text columns.")
                else:
                    with st.spinner("Processing rolling context..."):
                        try:
                            # Show progress
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("Preparing data...")
                            progress_bar.progress(25)
                            
                            # Process the data
                            speaker_col_param = speaker_col if speaker_col != "(None)" else None
                            
                            status_text.text("Generating context...")
                            progress_bar.progress(50)
                            
                            context_df = process_rolling_context(
                                df=current_data,
                                id_col=id_col,
                                text_col=text_col,
                                window_size=window_size,
                                speaker_col=speaker_col_param,
                                selected_speakers=selected_speakers if selected_speakers else None,
                                include_future=include_future,
                                future_window=future_window,
                                context_separator=context_separator
                            )
                            
                            progress_bar.progress(75)
                            status_text.text("Finalizing results...")
                            
                            st.session_state.context_data = context_df
                            st.session_state.processing_complete = True
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Processing complete!")
                            
                            time.sleep(0.5)  # Brief pause to show completion
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.success("✅ Rolling context generated successfully!")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Error during processing: {e}")
                            st.exception(e)
    
    with tab3:
        if not st.session_state.processing_complete:
            st.warning("⚠️ Please process your data first from the Processing tab.")
        else:
            st.subheader("📊 Processing Results")
            
            context_df = st.session_state.context_data
            
            # Results summary
            st.subheader("📈 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows Processed", len(context_df))
            with col2:
                rows_with_context = context_df['Has_Context'].sum()
                st.metric("Rows with Context", rows_with_context)
            with col3:
                avg_context_length = context_df['Context_Length'].mean()
                st.metric("Avg Context Length", f"{avg_context_length:.1f}")
            with col4:
                max_context_length = context_df['Context_Length'].max()
                st.metric("Max Context Length", max_context_length)
            
            # Context length distribution
            if len(context_df) > 0:
                st.subheader("📊 Context Length Distribution")
                context_dist = context_df['Context_Length'].value_counts().sort_index()
                st.bar_chart(context_dist)
            
            # Results preview
            st.subheader("👀 Results Preview")
            
            # Column selection for display
            all_cols = list(context_df.columns)
            
            # Default columns to show
            default_cols = []
            for col in ['conversation_id', 'ID', 'id', 'Statement', 'Past_Context']:
                if col in all_cols:
                    default_cols.append(col)
            
            # If no default columns found, show first few
            if not default_cols:
                default_cols = all_cols[:4]
            
            selected_cols = st.multiselect(
                "Select columns to display:",
                options=all_cols,
                default=default_cols,
                help="Choose which columns to show in the preview"
            )
            
            if selected_cols:
                n_rows = st.slider("Number of rows to display:", 5, min(50, len(context_df)), 10)
                
                # Show the data
                st.dataframe(context_df[selected_cols].head(n_rows), height=400)
                
                # Show a specific example with context
                if len(context_df) > 0:
                    st.subheader("🔍 Context Example")
                    
                    # Find a row with good context
                    context_rows = context_df[context_df['Context_Length'] > 0]
                    if len(context_rows) > 0:
                        example_row = context_rows.iloc[0]
                        
                        st.write("**Current Statement:**")
                        st.write(f"*{example_row['Statement']}*")
                        
                        if example_row['Past_Context']:
                            st.write("**Past Context:**")
                            st.write(f"*{example_row['Past_Context']}*")
                        
                        if include_future and 'Future_Context' in example_row and example_row['Future_Context']:
                            st.write("**Future Context:**")
                            st.write(f"*{example_row['Future_Context']}*")
            
            # Download section
            st.subheader("💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Full results
                csv_full = context_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv_full,
                    file_name='rolling_context_full.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            
            with col2:
                # Essential columns only
                essential_cols = []
                for col in context_df.columns:
                    if any(key in col.lower() for key in ['id', 'statement', 'context', 'speaker']):
                        essential_cols.append(col)
                
                if essential_cols:
                    csv_essential = context_df[essential_cols].to_csv(index=False)
                    st.download_button(
                        label="📥 Download Essential Columns (CSV)",
                        data=csv_essential,
                        file_name='rolling_context_essential.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
            
            # Export configuration
            st.subheader("⚙️ Processing Summary")
            
            # Create summary of processing settings
            processing_summary = {
                'Setting': ['ID Column', 'Text Column', 'Speaker Column', 'Window Size', 'Future Context', 'Future Window', 'Context Separator', 'Selected Speakers'],
                'Value': [
                    getattr(st.session_state, 'id_col', 'N/A'),
                    getattr(st.session_state, 'text_col', 'N/A'),
                    getattr(st.session_state, 'speaker_col', 'None'),
                    getattr(st.session_state, 'window_size', 'N/A'),
                    'Yes' if include_future else 'No',
                    future_window if include_future else 'N/A',
                    repr(context_separator),
                    ', '.join(getattr(st.session_state, 'selected_speakers', [])) or 'All'
                ]
            }
            
            summary_df = pd.DataFrame(processing_summary)
            st.dataframe(summary_df, hide_index=True)

if __name__ == "__main__":
    main()
