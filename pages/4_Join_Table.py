import streamlit as st
import pandas as pd
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple
import numpy as np
import io

class TableJoiner:
    """
    Advanced CSV table joining with fuzzy matching and multiple join types.
    Supports inner, left, and right joins with optional fuzzy text matching.
    """
    
    def __init__(self):
        """Initialize the table joiner."""
        self.data1 = None
        self.data2 = None
        self.file1_name = ""
        self.file2_name = ""
        self.joined_data = None
        self.join_config = {
            'left_column': '',
            'right_column': '',
            'join_type': 'inner',
            'fuzzy_matching': False,
            'remove_duplicates': False,
            'fuzzy_threshold': 0.8
        }
    
    def load_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame, file1_name: str, file2_name: str):
        """
        Load two DataFrames for joining.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            file1_name: Name of first file
            file2_name: Name of second file
        """
        self.data1 = df1.copy()
        self.data2 = df2.copy()
        self.file1_name = file1_name
        self.file2_name = file2_name
    
    def set_join_config(self, left_column: str, right_column: str, 
                       join_type: str = 'inner', fuzzy_matching: bool = False, 
                       remove_duplicates: bool = False, fuzzy_threshold: float = 0.8):
        """Configure join parameters."""
        self.join_config = {
            'left_column': left_column,
            'right_column': right_column,
            'join_type': join_type,
            'fuzzy_matching': fuzzy_matching,
            'remove_duplicates': remove_duplicates,
            'fuzzy_threshold': fuzzy_threshold
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for fuzzy matching."""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove punctuation and extra spaces
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def fuzzy_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        if norm1 == norm2:
            return 1.0
        
        # Use SequenceMatcher for similarity calculation
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Also check for word overlap
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if words1 and words2:
            word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            # Take the maximum of sequence similarity and word overlap
            similarity = max(similarity, word_overlap)
        
        return similarity
    
    def find_fuzzy_match(self, target_value: str, candidates: pd.Series) -> Tuple[Optional[int], float]:
        """Find the best fuzzy match for a target value in a series of candidates."""
        best_match_idx = None
        best_similarity = 0.0
        
        for idx, candidate in candidates.items():
            similarity = self.fuzzy_similarity(target_value, candidate)
            if similarity > best_similarity and similarity >= self.join_config['fuzzy_threshold']:
                best_similarity = similarity
                best_match_idx = idx
        
        return best_match_idx, best_similarity
    
    def perform_join(self) -> pd.DataFrame:
        """Perform the table join operation."""
        if self.data1 is None or self.data2 is None:
            raise ValueError("Please load both files first")
        
        left_col = self.join_config['left_column']
        right_col = self.join_config['right_column']
        join_type = self.join_config['join_type']
        fuzzy_matching = self.join_config['fuzzy_matching']
        
        if not left_col or not right_col:
            raise ValueError("Please specify both join columns")
        
        if fuzzy_matching:
            result = self._perform_fuzzy_join()
        else:
            result = self._perform_exact_join()
        
        # Remove duplicates if requested
        if self.join_config['remove_duplicates']:
            result = result.drop_duplicates()
        
        self.joined_data = result
        return result
    
    def _perform_exact_join(self) -> pd.DataFrame:
        """Perform exact matching join using pandas merge."""
        left_col = self.join_config['left_column']
        right_col = self.join_config['right_column']
        join_type = self.join_config['join_type']
        
        # Map join types to pandas merge types
        how_map = {
            'inner': 'inner',
            'left': 'left',
            'right': 'right'
        }
        
        return pd.merge(
            self.data1, 
            self.data2, 
            left_on=left_col, 
            right_on=right_col, 
            how=how_map[join_type],
            suffixes=('_left', '_right')
        )
    
    def _perform_fuzzy_join(self) -> pd.DataFrame:
        """Perform fuzzy matching join."""
        left_col = self.join_config['left_column']
        right_col = self.join_config['right_column']
        join_type = self.join_config['join_type']
        
        result_rows = []
        
        if join_type in ['inner', 'left']:
            # Process all rows from left table
            for idx1, row1 in self.data1.iterrows():
                target_value = row1[left_col]
                
                # Find best fuzzy match in right table
                match_idx, similarity = self.find_fuzzy_match(target_value, self.data2[right_col])
                
                if match_idx is not None:
                    # Found a match
                    row2 = self.data2.loc[match_idx]
                    combined_row = {**row1.to_dict(), **row2.to_dict()}
                    combined_row['_match_similarity'] = similarity
                    result_rows.append(combined_row)
                elif join_type == 'left':
                    # No match found but include row (left join)
                    combined_row = row1.to_dict()
                    # Add empty columns from right table
                    for col in self.data2.columns:
                        if col not in combined_row:
                            combined_row[col] = np.nan
                    combined_row['_match_similarity'] = 0.0
                    result_rows.append(combined_row)
        
        if join_type == 'right':
            # Process all rows from right table
            for idx2, row2 in self.data2.iterrows():
                target_value = row2[right_col]
                
                # Find best fuzzy match in left table
                match_idx, similarity = self.find_fuzzy_match(target_value, self.data1[left_col])
                
                if match_idx is not None:
                    # Found a match
                    row1 = self.data1.loc[match_idx]
                    combined_row = {**row1.to_dict(), **row2.to_dict()}
                    combined_row['_match_similarity'] = similarity
                    result_rows.append(combined_row)
                else:
                    # No match found but include row (right join)
                    combined_row = row2.to_dict()
                    # Add empty columns from left table
                    for col in self.data1.columns:
                        if col not in combined_row:
                            combined_row[col] = np.nan
                    combined_row['_match_similarity'] = 0.0
                    result_rows.append(combined_row)
        
        return pd.DataFrame(result_rows)
    
    def get_join_stats(self) -> Dict:
        """Get statistics about the join operation."""
        if self.joined_data is None:
            return {}
        
        stats = {
            'original_rows_1': len(self.data1),
            'original_rows_2': len(self.data2),
            'joined_rows': len(self.joined_data),
            'total_columns': len(self.joined_data.columns),
            'null_counts': self.joined_data.isnull().sum().to_dict(),
            'duplicates': self.joined_data.duplicated().sum()
        }
        
        # Add fuzzy matching stats if applicable
        if self.join_config['fuzzy_matching'] and '_match_similarity' in self.joined_data.columns:
            similarities = self.joined_data['_match_similarity']
            matches = similarities[similarities > 0]
            stats['fuzzy_stats'] = {
                'matches_found': len(matches),
                'avg_similarity': matches.mean() if len(matches) > 0 else 0,
                'min_similarity': matches.min() if len(matches) > 0 else 0,
                'max_similarity': matches.max() if len(matches) > 0 else 0
            }
        
        return stats

def main():
    st.set_page_config(
        page_title="CSV Table Joiner",
        page_icon="🔗",
        layout="wide"
    )
    
    st.title("🔗 CSV Table Joiner")
    st.markdown("**Advanced CSV table joining with fuzzy matching and multiple join types**")
    
    # Initialize session state
    if 'joiner' not in st.session_state:
        st.session_state.joiner = TableJoiner()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'join_completed' not in st.session_state:
        st.session_state.join_completed = False
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Upload First CSV File")
        file1 = st.file_uploader("Choose first CSV file", type=['csv'], key='file1')
        
        if file1 is not None:
            try:
                df1 = pd.read_csv(file1)
                st.success(f"✅ Loaded: {len(df1)} rows, {len(df1.columns)} columns")
                st.write("**Preview:**")
                st.dataframe(df1.head(), height=200)
                st.write("**Columns:**", list(df1.columns))
                
                # Store in session state
                st.session_state.df1 = df1
                st.session_state.file1_name = file1.name
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with col2:
        st.subheader("📁 Upload Second CSV File")
        file2 = st.file_uploader("Choose second CSV file", type=['csv'], key='file2')
        
        if file2 is not None:
            try:
                df2 = pd.read_csv(file2)
                st.success(f"✅ Loaded: {len(df2)} rows, {len(df2.columns)} columns")
                st.write("**Preview:**")
                st.dataframe(df2.head(), height=200)
                st.write("**Columns:**", list(df2.columns))
                
                # Store in session state
                st.session_state.df2 = df2
                st.session_state.file2_name = file2.name
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # Load data into joiner when both files are uploaded
    if hasattr(st.session_state, 'df1') and hasattr(st.session_state, 'df2'):
        if not st.session_state.data_loaded:
            st.session_state.joiner.load_dataframes(
                st.session_state.df1, 
                st.session_state.df2, 
                st.session_state.file1_name, 
                st.session_state.file2_name
            )
            st.session_state.data_loaded = True
        
        st.divider()
        
        # Join configuration
        st.subheader("🔧 Join Configuration")
        
        config_col1, config_col2, config_col3 = st.columns(3)
        
        with config_col1:
            # Column selection
            left_column = st.selectbox(
                "Select column from first table:",
                options=list(st.session_state.joiner.data1.columns),
                key='left_col'
            )
        
        with config_col2:
            right_column = st.selectbox(
                "Select column from second table:",
                options=list(st.session_state.joiner.data2.columns),
                key='right_col'
            )
        
        with config_col3:
            join_type = st.selectbox(
                "Join type:",
                options=['inner', 'left', 'right'],
                help="Inner: only matching rows, Left: all rows from first table, Right: all rows from second table"
            )
        
        # Advanced options
        with st.expander("🔍 Advanced Options"):
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                fuzzy_matching = st.checkbox(
                    "Enable fuzzy text matching",
                    help="Use similarity algorithms to match text that isn't exactly identical"
                )
                
                if fuzzy_matching:
                    fuzzy_threshold = st.slider(
                        "Fuzzy matching threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.8,
                        step=0.1,
                        help="Minimum similarity score to consider a match (0.0 = very loose, 1.0 = exact)"
                    )
                else:
                    fuzzy_threshold = 0.8
            
            with adv_col2:
                remove_duplicates = st.checkbox(
                    "Remove duplicate rows",
                    help="Remove duplicate rows from the final result"
                )
        
        # Perform join
        if st.button("🚀 Perform Join", type="primary"):
            try:
                with st.spinner("Performing join operation..."):
                    st.session_state.joiner.set_join_config(
                        left_column=left_column,
                        right_column=right_column,
                        join_type=join_type,
                        fuzzy_matching=fuzzy_matching,
                        remove_duplicates=remove_duplicates,
                        fuzzy_threshold=fuzzy_threshold
                    )
                    
                    result = st.session_state.joiner.perform_join()
                    st.session_state.join_completed = True
                    
                st.success("✅ Join completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during join: {e}")
    
    # Display results
    if st.session_state.join_completed and st.session_state.joiner.joined_data is not None:
        st.divider()
        st.subheader("📊 Join Results")
        
        # Statistics
        stats = st.session_state.joiner.get_join_stats()
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Original Rows (Table 1)", stats['original_rows_1'])
        with stat_col2:
            st.metric("Original Rows (Table 2)", stats['original_rows_2'])
        with stat_col3:
            st.metric("Joined Rows", stats['joined_rows'])
        with stat_col4:
            st.metric("Total Columns", stats['total_columns'])
        
        # Fuzzy matching statistics
        if 'fuzzy_stats' in stats:
            st.subheader("🎯 Fuzzy Matching Statistics")
            fuzz_col1, fuzz_col2, fuzz_col3, fuzz_col4 = st.columns(4)
            
            with fuzz_col1:
                st.metric("Matches Found", stats['fuzzy_stats']['matches_found'])
            with fuzz_col2:
                st.metric("Avg Similarity", f"{stats['fuzzy_stats']['avg_similarity']:.3f}")
            with fuzz_col3:
                st.metric("Min Similarity", f"{stats['fuzzy_stats']['min_similarity']:.3f}")
            with fuzz_col4:
                st.metric("Max Similarity", f"{stats['fuzzy_stats']['max_similarity']:.3f}")
        
        # Data preview
        st.subheader("📋 Data Preview")
        
        # Show number of rows to display
        n_rows = st.slider("Number of rows to display:", 5, min(50, len(st.session_state.joiner.joined_data)), 10)
        
        st.dataframe(st.session_state.joiner.joined_data.head(n_rows), height=400)
        
        # Column information
        with st.expander("📝 Column Information"):
            st.write("**All Columns:**")
            st.write(list(st.session_state.joiner.joined_data.columns))
            
            # Null value analysis
            null_counts = {col: count for col, count in stats['null_counts'].items() if count > 0}
            if null_counts:
                st.write("**Columns with null values:**")
                for col, count in null_counts.items():
                    percentage = (count / stats['joined_rows']) * 100
                    st.write(f"• {col}: {count} nulls ({percentage:.1f}%)")
        
        # Download results
        st.subheader("💾 Download Results")
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        st.session_state.joiner.joined_data.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"joined_data_{join_type}_join.csv",
            mime="text/csv",
            help="Download the joined data as a CSV file"
        )
        
        # Show duplicate information
        if stats['duplicates'] > 0:
            st.warning(f"⚠️ Found {stats['duplicates']} duplicate rows in the result")

if __name__ == "__main__":
    main()
