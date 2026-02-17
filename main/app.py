# CSV File Upload and Display App with Column Selection Options
import streamlit as st
import pandas as pd
import numpy as np

# Set page title
st.title("📊 CSV File Viewer with Column Options")

# File uploader with custom message
uploaded_file = st.file_uploader("Upload your CSV file here", type=['csv'])

# Initialize session state for processed DataFrame
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

# Initialize session state for showing updated data
if 'show_updated_data' not in st.session_state:
    st.session_state.show_updated_data = False

# Initialize session state for last operation details (for showing in preview)
if 'last_operation_details' not in st.session_state:
    st.session_state.last_operation_details = None

# Reset show_updated_data when a new file is uploaded
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

# Check if a new file was uploaded - reset states if so
if uploaded_file is not None:
    if st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.show_updated_data = False
        st.session_state.last_uploaded_file = uploaded_file.name

# Initialize session state for filters history
if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = []

# ==========================================
# UNDO/REDO INITIALIZATION
# ==========================================
# Initialize undo stack to store previous states
if 'undo_stack' not in st.session_state:
    st.session_state.undo_stack = []

# Initialize redo stack to store undone states
if 'redo_stack' not in st.session_state:
    st.session_state.redo_stack = []

# Initialize operation counter for tracking
if 'operation_count' not in st.session_state:
    st.session_state.operation_count = 0

# Initialize undo/redo operation counters
if 'undo_count' not in st.session_state:
    st.session_state.undo_count = 0

if 'redo_count' not in st.session_state:
    st.session_state.redo_count = 0

# Maximum number of undo/redo operations allowed
MAX_UNDO_REDO_OPERATIONS = 20

# Maximum number of undo steps to maintain
MAX_UNDO_STEPS = 20

def save_state_before_operation(operation_name):
    """Save current state to undo stack before performing an operation."""
    if st.session_state.processed_df is not None:
        # Only save state if it's different from the last saved state
        undo_stack = st.session_state.undo_stack
        if undo_stack:
            last_state = undo_stack[-1]
            if (last_state['df'].equals(st.session_state.processed_df) and 
                last_state['filters'] == st.session_state.filters_applied):
                # State hasn't changed, don't duplicate
                return
        
        # Add current state to undo stack
        state = {
            'df': st.session_state.processed_df.copy(),
            'filters': list(st.session_state.filters_applied),  # Create a copy of filters
            'operation': operation_name,
            'count': st.session_state.operation_count
        }
        st.session_state.undo_stack.append(state)
        st.session_state.redo_stack = []  # Clear redo stack on new operation
        st.session_state.operation_count += 1
        st.session_state.undo_count = 0  # Reset undo count on new operation
        st.session_state.redo_count = 0  # Reset redo count on new operation
        
        # Limit undo stack size
        if len(st.session_state.undo_stack) > MAX_UNDO_STEPS:
            st.session_state.undo_stack.pop(0)

def undo():
    """Perform undo operation - restore previous state."""
    # Check if undo limit is reached
    if st.session_state.undo_count >= MAX_UNDO_REDO_OPERATIONS:
        st.warning(f"⚠️ Maximum undo limit ({MAX_UNDO_REDO_OPERATIONS}) reached!")
        return False
    
    if st.session_state.undo_stack:
        # Save current state to redo stack
        current_state = {
            'df': st.session_state.processed_df.copy(),
            'filters': list(st.session_state.filters_applied),
            'operation': "Current state",
            'count': st.session_state.operation_count
        }
        st.session_state.redo_stack.append(current_state)
        
        # Restore previous state
        previous_state = st.session_state.undo_stack.pop()
        st.session_state.processed_df = previous_state['df']
        st.session_state.filters_applied = previous_state['filters']
        st.session_state.operation_count = previous_state['count']
        st.session_state.undo_count += 1  # Increment undo count
        st.session_state.redo_count = 0  # Reset redo count on new undo
        return True
    return False

def redo():
    """Perform redo operation - restore next state."""
    # Check if redo limit is reached
    if st.session_state.redo_count >= MAX_UNDO_REDO_OPERATIONS:
        st.warning(f"⚠️ Maximum redo limit ({MAX_UNDO_REDO_OPERATIONS}) reached!")
        return False
    
    if st.session_state.redo_stack:
        # Save current state to undo stack
        current_state = {
            'df': st.session_state.processed_df.copy(),
            'filters': list(st.session_state.filters_applied),
            'operation': "Current state",
            'count': st.session_state.operation_count
        }
        st.session_state.undo_stack.append(current_state)
        
        # Restore next state
        next_state = st.session_state.redo_stack.pop()
        st.session_state.processed_df = next_state['df']
        st.session_state.filters_applied = next_state['filters']
        st.session_state.operation_count = next_state['count']
        st.session_state.redo_count += 1  # Increment redo count
        st.session_state.undo_count = 0  # Reset undo count on new redo
        return True
    return False

def clear_redo_stack():
    """Clear the redo stack when a new operation is performed."""
    st.session_state.redo_stack = []

# ==========================================
# SIDEBAR - UNDO/REDO QUICK ACCESS
# ==========================================
with st.sidebar:
    st.header("🔄 Undo/Redo History")
    
    # Undo/Redo buttons with icons
    col_undo, col_redo = st.columns(2)
    
    with col_undo:
        undo_disabled = len(st.session_state.undo_stack) == 0 or st.session_state.undo_count >= MAX_UNDO_REDO_OPERATIONS
        undo_tooltip = "No more undo operations available" if st.session_state.undo_count >= MAX_UNDO_REDO_OPERATIONS else "Undo last operation"
        if st.button("↩️ Undo", disabled=undo_disabled, use_container_width=True, help=undo_tooltip):
            if undo():
                st.success("↩️ Undo successful!")
                st.rerun()
            else:
                st.error("Nothing to undo")
    
    with col_redo:
        redo_disabled = len(st.session_state.redo_stack) == 0 or st.session_state.redo_count >= MAX_UNDO_REDO_OPERATIONS
        redo_tooltip = "No more redo operations available" if st.session_state.redo_count >= MAX_UNDO_REDO_OPERATIONS else "Redo last operation"
        if st.button("↪️ Redo", disabled=redo_disabled, use_container_width=True, help=redo_tooltip):
            if redo():
                st.success("↪️ Redo successful!")
                st.rerun()
            else:
                st.error("Nothing to redo")
    
    # Show history statistics
    st.divider()
    st.caption(f"📊 **History Stats:**")
    st.info(f"↩️ Undo: **{st.session_state.undo_count}/{MAX_UNDO_REDO_OPERATIONS}** used")
    st.info(f"↪️ Redo: **{st.session_state.redo_count}/{MAX_UNDO_REDO_OPERATIONS}** used")
    st.info(f"📈 Total operations: **{st.session_state.operation_count}**")
    st.info(f"📦 Saved states: **{len(st.session_state.undo_stack)}/{MAX_UNDO_STEPS}**")
    
    # Show operation history
    if st.session_state.undo_stack:
        st.divider()
        st.caption("📜 **Recent Operations:**")
        # Show last 6 operations
        recent_ops = list(reversed(st.session_state.undo_stack))[-6:]
        for i, state in enumerate(recent_ops, 1):
            op_num = len(st.session_state.undo_stack) - len(recent_ops) + i
            st.write(f"{op_num}. {state['operation']}")
    
    # Clear all history button
    st.divider()
    if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
        st.session_state.undo_stack = []
        st.session_state.redo_stack = []
        st.session_state.operation_count = 0
        st.success("History cleared!")
        st.rerun()
    
    st.divider()
    st.caption("💡 **Tip:** Use the undo/redo buttons above to quickly revert or reapply data operations at any time!")

# Check if file is uploaded
if uploaded_file is not None:
    try:
        # Read the CSV file as DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Initialize processed_df with original df if not set
        if st.session_state.processed_df is None:
            st.session_state.processed_df = df.copy()
        
        # Display success message
        st.success("✅ File uploaded successfully!")
        
        # Display the DataFrame
        st.subheader("📋 Data Preview:")
        st.dataframe(df)
        
        # Show basic info about the DataFrame
        st.subheader("📊 DataFrame Info:")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        with col2:
            st.info(f"**Columns:** {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
        
        st.divider()
        
        # ==========================================
        # FILTER OPTIONS SECTION
        # ==========================================
        st.divider()
        st.subheader("🔧 Filter Options")
        st.write("Apply various operations to clean and filter your data:")
        
        # Create a copy of the current processed_df for applying filters
        filtered_df = st.session_state.processed_df.copy()
        
        # Store original shape
        original_shape = filtered_df.shape
        
        # Show filters applied so far
        if st.session_state.filters_applied:
            st.info(f"📌 **Filters Applied ({len(st.session_state.filters_applied)}):**")
            for i, f in enumerate(st.session_state.filters_applied, 1):
                st.write(f"  {i}. {f}")
        
        # 1. DROP NULL VALUES
        with st.expander("🗑️ Drop Null Values", expanded=False):
            col_drop_null_1, col_drop_null_2 = st.columns(2)
            with col_drop_null_1:
                drop_rows_null = st.checkbox("Drop rows with null values", value=False, 
                    help="Removes any row that contains at least one null/NaN value")
            with col_drop_null_2:
                drop_cols_null = st.checkbox("Drop columns with null values", value=False,
                    help="Removes any column that contains at least one null/NaN value")
            
            # Auto-apply drop null values immediately
            if drop_rows_null or drop_cols_null:
                # Save state before modification for undo functionality
                save_state_before_operation("Drop Null Values")
                
                rows_before = len(filtered_df)
                cols_before = len(filtered_df.columns)
                
                if drop_rows_null:
                    filtered_df = filtered_df.dropna()
                if drop_cols_null:
                    filtered_df = filtered_df.dropna(axis=1)
                
                rows_after = len(filtered_df)
                cols_after = len(filtered_df.columns)
                
                # Store operation details for show updated data feature
                st.session_state.last_operation_details = {
                    'operation': 'Drop Null Values',
                    'drop_rows': drop_rows_null,
                    'drop_cols': drop_cols_null,
                    'rows_before': rows_before,
                    'rows_after': rows_after,
                    'cols_before': cols_before,
                    'cols_after': cols_after,
                    'rows_removed': rows_before - rows_after,
                    'cols_removed': cols_before - cols_after
                }
                
                # Apply changes immediately to processed_df
                st.session_state.processed_df = filtered_df.copy()
                
                st.success(f"✅ Dropped {rows_before - rows_after} rows and {cols_before - cols_after} columns with null values")
                st.write(f"**New Shape:** {filtered_df.shape}")
                st.rerun()
        
        # 2. DROP COLUMN
        with st.expander("🗑️ Drop Column", expanded=False):
            cols_to_drop = st.multiselect(
                "Select columns to drop:",
                filtered_df.columns,
                help="Select one or more columns to remove from the dataset"
            )
            
            # Auto-apply drop column immediately
            if cols_to_drop:
                # Save state before modification for undo functionality
                save_state_before_operation(f"Drop Column(s): {', '.join(cols_to_drop)}")
                
                cols_before = len(filtered_df.columns)
                filtered_df = filtered_df.drop(columns=cols_to_drop)
                cols_after = len(filtered_df.columns)
                
                # Store operation details for show updated data feature
                st.session_state.last_operation_details = {
                    'operation': 'Drop Column(s)',
                    'columns_dropped': cols_to_drop,
                    'cols_before': cols_before,
                    'cols_after': cols_after,
                    'cols_removed': cols_before - cols_after
                }
                
                # Apply changes immediately to processed_df
                st.session_state.processed_df = filtered_df.copy()
                
                st.success(f"✅ Dropped {cols_before - cols_after} column(s): {', '.join(cols_to_drop)}")
                st.write(f"**New Shape:** {filtered_df.shape}")
                st.rerun()
        
        # 3. DROP DUPLICATE VALUES
        with st.expander("🔄 Drop Duplicate Values", expanded=False):
            dup_subset = st.multiselect(
                "Select columns to check for duplicates (leave empty for all columns):",
                filtered_df.columns,
                help="Duplicates are rows that have identical values in all selected columns"
            )
            
            dup_keep = st.selectbox(
                "Keep:",
                ["first", "last", False],
                format_func=lambda x: "Keep first occurrence" if x == "first" else ("Keep last occurrence" if x == "last" else "Remove all duplicates"),
                help="'Keep first' keeps the first occurrence of each duplicate, 'Keep last' keeps the last, 'Remove all' removes all duplicates"
            )
            
            if st.button("Remove Duplicates", type="secondary"):
                # Save state before modification for undo functionality
                save_state_before_operation("Drop Duplicate Values")
                
                rows_before = len(filtered_df)
         
                if dup_subset:
                    filtered_df = filtered_df.drop_duplicates(subset=dup_subset, keep=dup_keep)
                else:
                    filtered_df = filtered_df.drop_duplicates(keep=dup_keep)
                rows_after = len(filtered_df)
                
                # Store operation details for show updated data feature
                st.session_state.last_operation_details = {
                    'operation': 'Drop Duplicate Values',
                    'subset': dup_subset,
                    'keep': dup_keep,
                    'rows_before': rows_before,
                    'rows_after': rows_after,
                    'rows_removed': rows_before - rows_after
                }
                
                # Apply changes immediately to processed_df
                st.session_state.processed_df = filtered_df.copy()
                
                st.success(f"✅ Removed {rows_before - rows_after} duplicate rows")
                st.write(f"**New Shape:** {filtered_df.shape}")
                st.rerun()
        
        # 4. REMOVE OUTLIERS
        with st.expander("📊 Remove Outliers", expanded=False):
            numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_columns:
                outlier_col = st.selectbox(
                    "Select column to check for outliers:",
                    numeric_columns,
                    help="Outliers are detected using the IQR (Interquartile Range) method"
                )
                
                outlier_multiplier = st.slider(
                    "IQR Multiplier:",
                    min_value=1.0,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="Standard multiplier is 1.5. Use 3.0 for more conservative outlier removal"
                )
                
                # Auto-apply remove outliers immediately
                if outlier_col and st.button("Remove Outliers", type="secondary"):
                    # Save state before modification for undo functionality
                    save_state_before_operation(f"Remove Outliers from '{outlier_col}'")
                    
                    rows_before = len(filtered_df)
                    
                    Q1 = filtered_df[outlier_col].quantile(0.25)
                    Q3 = filtered_df[outlier_col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - outlier_multiplier * IQR
                    upper_bound = Q3 + outlier_multiplier * IQR
                    
                    st.write(f"**Outlier Detection for '{outlier_col}':**")
                    st.write(f"- Q1 (25th percentile): {Q1:.2f}")
                    st.write(f"- Q3 (75th percentile): {Q3:.2f}")
                    st.write(f"- IQR: {IQR:.2f}")
                    st.write(f"- Lower bound: {lower_bound:.2f}")
                    st.write(f"- Upper bound: {upper_bound:.2f}")
                    
                    filtered_df = filtered_df[
                        (filtered_df[outlier_col] >= lower_bound) & 
                        (filtered_df[outlier_col] <= upper_bound)
                    ]
                    
                    rows_after = len(filtered_df)
                    
                    # Store operation details for show updated data feature
                    st.session_state.last_operation_details = {
                        'operation': 'Remove Outliers',
                        'column': outlier_col,
                        'Q1': Q1,
                        'Q3': Q3,
                        'IQR': IQR,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'rows_before': rows_before,
                        'rows_after': rows_after,
                        'rows_removed': rows_before - rows_after
                    }
                    
                    # Apply changes immediately to processed_df
                    st.session_state.processed_df = filtered_df.copy()
                    
                    st.success(f"✅ Removed {rows_before - rows_after} outlier rows from '{outlier_col}'")
                    st.write(f"**New Shape:** {filtered_df.shape}")
                    st.rerun()
            else:
                st.warning("⚠️ No numeric columns found in the dataset")
        
        # 5. DROP ROWS
        with st.expander("🗑️ Drop Rows", expanded=False):
            drop_rows_col = st.selectbox(
                "Select column to filter rows for removal:",
                filtered_df.columns,
                help="Choose a column to identify rows to drop"
            )
            
            if drop_rows_col:
                if pd.api.types.is_numeric_dtype(filtered_df[drop_rows_col]):
                    row_min = float(filtered_df[drop_rows_col].min())
                    row_max = float(filtered_df[drop_rows_col].max())
                    row_range = st.slider(
                        f"Select range of rows to keep (in {drop_rows_col}):",
                        row_min, row_max, (row_min, row_max),
                        help="Rows outside this range will be dropped"
                    )
                    
                    # Auto-apply drop rows by range immediately
                    if st.button("Apply Range Filter", type="secondary"):
                        # Save state before modification for undo functionality
                        save_state_before_operation(f"Drop Rows by Range in '{drop_rows_col}'")
                        
                        rows_before = len(filtered_df)
                        filtered_df = filtered_df[
                            (filtered_df[drop_rows_col] >= row_range[0]) & 
                            (filtered_df[drop_rows_col] <= row_range[1])
                        ]
                        rows_after = len(filtered_df)
                        
                        # Store operation details for show updated data feature
                        st.session_state.last_operation_details = {
                            'operation': 'Drop Rows by Range',
                            'column': drop_rows_col,
                            'range': row_range,
                            'rows_before': rows_before,
                            'rows_after': rows_after,
                            'rows_removed': rows_before - rows_after
                        }
                        
                        # Apply changes immediately to processed_df
                        st.session_state.processed_df = filtered_df.copy()
                        
                        st.success(f"✅ Kept {rows_after} rows (removed {rows_before - rows_after} rows outside range)")
                        st.write(f"**New Shape:** {filtered_df.shape}")
                        st.rerun()
                else:
                    row_values = filtered_df[drop_rows_col].unique()
                    rows_to_drop = st.multiselect(
                        f"Select values in '{drop_rows_col}' to remove:",
                        row_values,
                        help="All rows with selected values will be removed"
                    )
                    
                    # Auto-apply drop rows by values immediately
                    if rows_to_drop and st.button("Remove Selected Rows", type="secondary"):
                        # Save state before modification for undo functionality
                        save_state_before_operation(f"Drop Rows with '{drop_rows_col}' values")
                        
                        rows_before = len(filtered_df)
                        filtered_df = filtered_df[~filtered_df[drop_rows_col].isin(rows_to_drop)]
                        rows_after = len(filtered_df)
                        
                        # Store operation details for show updated data feature
                        st.session_state.last_operation_details = {
                            'operation': 'Drop Rows by Values',
                            'column': drop_rows_col,
                            'values_dropped': rows_to_drop,
                            'rows_before': rows_before,
                            'rows_after': rows_after,
                            'rows_removed': rows_before - rows_after
                        }
                        
                        # Apply changes immediately to processed_df
                        st.session_state.processed_df = filtered_df.copy()
                        
                        st.success(f"✅ Removed {rows_before - rows_after} rows")
                        st.write(f"**New Shape:** {filtered_df.shape}")
                        st.rerun()
        
        # 6. MULTIPLE FILTERS
        with st.expander("🔍 Multiple Filters", expanded=False):
            st.write("Apply multiple filters to your data:")
            
            # Filter 1
            col_filter1_1, col_filter1_2, col_filter1_3 = st.columns(3)
            with col_filter1_1:
                filter1_col = st.selectbox("Filter 1 - Column:", filtered_df.columns, key="filter1_col")
            with col_filter1_2:
                filter1_op = st.selectbox("Filter 1 - Operator:", 
                    ["==", "!=", ">", "<", ">=", "<=", "contains", "not contains"], key="filter1_op")
            with col_filter1_3:
                filter1_val = st.text_input("Filter 1 - Value:", key="filter1_val")
            
            # Filter 2
            col_filter2_1, col_filter2_2, col_filter2_3 = st.columns(3)
            with col_filter2_1:
                filter2_col = st.selectbox("Filter 2 - Column:", filtered_df.columns, key="filter2_col")
            with col_filter2_2:
                filter2_op = st.selectbox("Filter 2 - Operator:", 
                    ["==", "!=", ">", "<", ">=", "<=", "contains", "not contains"], key="filter2_op")
            with col_filter2_3:
                filter2_val = st.text_input("Filter 2 - Value:", key="filter2_val")
            
            # Filter 3
            col_filter3_1, col_filter3_2, col_filter3_3 = st.columns(3)
            with col_filter3_1:
                filter3_col = st.selectbox("Filter 3 - Column:", filtered_df.columns, key="filter3_col")
            with col_filter3_2:
                filter3_op = st.selectbox("Filter 3 - Operator:", 
                    ["==", "!=", ">", "<", ">=", "<=", "contains", "not contains"], key="filter3_op")
            with col_filter3_3:
                filter3_val = st.text_input("Filter 3 - Value:", key="filter3_val")
            
            # Filter combination logic
            filter_logic = st.radio(
                "Combine filters with:",
                ["AND (all conditions must be true)", "OR (at least one condition must be true)"],
                help="AND requires all filters to match, OR requires at least one filter to match"
            )
            filter_is_and = "AND" in filter_logic
            
            # Auto-apply multiple filters immediately
            if st.button("Apply Multiple Filters", type="secondary"):
                try:
                    # Save state before modification for undo functionality
                    save_state_before_operation("Apply Multiple Filters")
                    
                    # Create a mask for filtering
                    masks = []
                    
                    # Process each filter
                    for col, op, val in [(filter1_col, filter1_op, filter1_val), 
                                         (filter2_col, filter2_op, filter2_val), 
                                         (filter3_col, filter3_op, filter3_val)]:
                        if col and op and val:
                            # Convert value to appropriate type
                            col_dtype = filtered_df[col].dtype
                            if pd.api.types.is_numeric_dtype(col_dtype):
                                try:
                                    val = float(val)
                                except ValueError:
                                    pass
                            
                            # Create mask based on operator
                            if op == "==":
                                mask = filtered_df[col] == val
                            elif op == "!=":
                                mask = filtered_df[col] != val
                            elif op == ">":
                                mask = filtered_df[col] > val
                            elif op == "<":
                                mask = filtered_df[col] < val
                            elif op == ">=":
                                mask = filtered_df[col] >= val
                            elif op == "<=":
                                mask = filtered_df[col] <= val
                            elif op == "contains":
                                mask = filtered_df[col].astype(str).str.contains(str(val), case=False, na=False)
                            elif op == "not contains":
                                mask = ~filtered_df[col].astype(str).str.contains(str(val), case=False, na=False)
                            masks.append(mask)
                    
                    if masks:
                        rows_before = len(filtered_df)
                        
                        if filter_is_and:
                            combined_mask = masks[0] & masks[1] & masks[2] if len(masks) == 3 else \
                                           (masks[0] & masks[1] if len(masks) == 2 else masks[0])
                        else:
                            combined_mask = masks[0] | masks[1] | masks[2] if len(masks) == 3 else \
                                           (masks[0] | masks[1] if len(masks) == 2 else masks[0])
                        
                        filtered_df = filtered_df[combined_mask]
                        rows_after = len(filtered_df)
                        
                        # Store operation details for show updated data feature
                        filters_applied = []
                        for col, op, val in [(filter1_col, filter1_op, filter1_val), 
                                             (filter2_col, filter2_op, filter2_val), 
                                             (filter3_col, filter3_op, filter3_val)]:
                            if col and op and val:
                                filters_applied.append(f"{col} {op} {val}")
                        
                        st.session_state.last_operation_details = {
                            'operation': 'Multiple Filters',
                            'filters': filters_applied,
                            'logic': 'AND' if filter_is_and else 'OR',
                            'rows_before': rows_before,
                            'rows_after': rows_after,
                            'rows_removed': rows_before - rows_after
                        }
                        
                        # Apply changes immediately to processed_df
                        st.session_state.processed_df = filtered_df.copy()
                        
                        st.success(f"✅ Applied filters: {rows_before} → {rows_after} rows")
                        st.write(f"**New Shape:** {filtered_df.shape}")
                        st.rerun()
                    else:
                        st.warning("⚠️ Please set at least one filter")
                except Exception as e:
                    st.error(f"❌ Error applying filters: {e}")
        
        # ==========================================
        # DOWNLOAD AND SHOW UPDATED DATA
        # ==========================================
        st.divider()
        
        # Create columns for buttons
        col_download, col_show = st.columns([1, 1])
        
        with col_download:
            # Download processed data
            csv_processed = st.session_state.processed_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Processed Data",
                data=csv_processed,
                file_name="processed_data.csv",
                mime="text/csv"
            )
        
        with col_show:
            # Show Updated Data button
            if st.button("📊 Show Updated Data", use_container_width=True):
                st.session_state.show_updated_data = not st.session_state.show_updated_data
                st.rerun()
        
        # Display updated data if the flag is set
        if st.session_state.show_updated_data and st.session_state.processed_df is not None:
            st.divider()
            st.subheader("📊 Updated Data Preview:")
            st.dataframe(st.session_state.processed_df)
            st.success(f"✅ Showing {len(st.session_state.processed_df)} rows and {len(st.session_state.processed_df.columns)} columns")
            
            # Show operation details if available
            if st.session_state.last_operation_details:
                details = st.session_state.last_operation_details
                st.divider()
                st.subheader(f"📈 Last Operation: {details['operation']}")
                
                if details['operation'] == 'Remove Outliers':
                    st.write(f"**Column:** {details['column']}")
                    st.write(f"- Q1 (25th percentile): {details['Q1']:.2f}")
                    st.write(f"- Q3 (75th percentile): {details['Q3']:.2f}")
                    st.write(f"- IQR: {details['IQR']:.2f}")
                    st.write(f"- Lower bound: {details['lower_bound']:.2f}")
                    st.write(f"- Upper bound: {details['upper_bound']:.2f}")
                    st.write(f"- Rows before: {details['rows_before']}")
                    st.write(f"- Rows after: {details['rows_after']}")
                    st.warning(f"⚠️ Removed {details['rows_removed']} outlier rows")
                
                elif details['operation'] == 'Drop Null Values':
                    if details.get('drop_rows'):
                        st.write(f"- Rows removed: {details['rows_removed']}")
                    if details.get('drop_cols'):
                        st.write(f"- Columns removed: {details['cols_removed']}")
                    st.write(f"- Rows before: {details['rows_before']} → after: {details['rows_after']}")
                    st.write(f"- Columns before: {details['cols_before']} → after: {details['cols_after']}")
                    st.warning(f"⚠️ Removed {details['rows_removed']} rows and {details['cols_removed']} columns with null values")
                
                elif details['operation'] == 'Drop Column(s)':
                    st.write(f"**Columns dropped:** {', '.join(details['columns_dropped'])}")
                    st.write(f"- Columns before: {details['cols_before']} → after: {details['cols_after']}")
                    st.warning(f"⚠️ Removed {details['cols_removed']} column(s)")
                
                elif details['operation'] == 'Drop Duplicate Values':
                    subset = details.get('subset')
                    keep = details.get('keep')
                    if subset:
                        st.write(f"- Checked columns: {', '.join(subset)}")
                    else:
                        st.write(f"- Checked columns: All columns")
                    st.write(f"- Keep option: {keep}")
                    st.write(f"- Rows before: {details['rows_before']} → after: {details['rows_after']}")
                    st.warning(f"⚠️ Removed {details['rows_removed']} duplicate rows")
                
                elif details['operation'] == 'Drop Rows by Range':
                    st.write(f"**Column:** {details['column']}")
                    st.write(f"- Range: {details['range'][0]} to {details['range'][1]}")
                    st.write(f"- Rows before: {details['rows_before']} → after: {details['rows_after']}")
                    st.warning(f"⚠️ Removed {details['rows_removed']} rows outside range")
                
                elif details['operation'] == 'Drop Rows by Values':
                    st.write(f"**Column:** {details['column']}")
                    st.write(f"- Values removed: {', '.join(map(str, details['values_dropped']))}")
                    st.write(f"- Rows before: {details['rows_before']} → after: {details['rows_after']}")
                    st.warning(f"⚠️ Removed {details['rows_removed']} rows")
                
                elif details['operation'] == 'Multiple Filters':
                    st.write(f"**Filters applied:**")
                    for f in details['filters']:
                        st.write(f"  - {f}")
                    st.write(f"- Logic: {details['logic']}")
                    st.write(f"- Rows before: {details['rows_before']} → after: {details['rows_after']}")
                    st.warning(f"⚠️ Removed {details['rows_removed']} rows based on filters")
            
            st.info("💡 You can now download this filtered data using the download button above.")
    
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

