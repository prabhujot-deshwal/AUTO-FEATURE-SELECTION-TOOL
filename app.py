import streamlit as st
from helpers import *

def main():
    st.title("AUTO FEATURE SELECTOR TOOL")

    ###file uploader
    uploaded_file = st.file_uploader("Choose a CSV file",type="csv")

    if uploaded_file is not None:
        analyze_csv_file(uploaded_file)

        df = st.session_state.get('df')

        if df is not None:
            st.write("### Encode Categorical Columns")
            encoding_method = st.selectbox("Choose encoding method", ["Label Encoding", "One-Hot Encoding"])

            if st.button("Transform Object Columns"):
                df, label_encoders = transform_object_columns(df, encoding_method)
                st.session_state['df'] = df
                st.success("Object columns transformed successfully")
                st.write("### Transformed DataFrame:")
                st.write(df)

            select_target_column(df)

            feature_selection_options()

            if st.session_state.get('feature_selection_method') == "Correlation Coefficient":
                correlation_cofficient_selection()

            if st.session_state.get('feature_selection_method') == "Chi-Square Test":
                chi_square_selection() 

            
            if st.session_state.get('feature_selection_method') == "ANOVA":
                anova_selection()

            if st.session_state.get('feature_selection_method') == "Mutual Information":
                mutual_information_selection()


            if st.session_state.get('feature_selection_method') == "Variance Threshold":
                variance_threshold_selection()
 
if __name__ == "__main__":
    main()