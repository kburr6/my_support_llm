# app.py
import streamlit as st
import os
import io
import uuid
import json
import pandas as pd
from datetime import datetime, time
import matplotlib.pyplot as plt
from streamlit.runtime.scriptrunner import get_script_run_ctx
from backend import (
    get_chains,
    add_document_to_db,
    list_all_documents,
    delete_document_by_id,
    get_all_curated_qa,
    add_curated_qa,
    delete_curated_qa,
    log_feedback,
    add_url_to_db,
    get_feedback_log,
    log_query,
    get_query_log,
    get_ai_config,
    update_ai_config,
    get_dashboard_kpis,    
    get_top_problem_questions,  
    get_top_information_gaps,   
    get_most_frequent_questions,
    reset_logs_for_question,
    add_slack_dir_to_db,
    add_directory_to_db,
    delete_all_data,
    LLM_MODEL
)


# --- Initialize Session State ---
if "question_to_curate" not in st.session_state:
    st.session_state.question_to_curate = ""

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Support AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Product Support AI Assistant")
st.markdown("Ask questions about your documents, manage your knowledge base, and get accurate answers.")

# --- NAVIGATION ---
TABS = [
    "💬 Chat with AI", "📈 Dashboard", "🗂️ Manage Documents",
    "📝 Curated Q&A", "📊 Feedback Log", "🔍 Query Inspector", "🤖 AI Config"
]
# Use a key to manage the radio button's state via st.session_state
if "active_tab_index" not in st.session_state:
    st.session_state.active_tab_index = 1 # Default to the Dashboard (index 1)

st.sidebar.title("Navigation")
# The 'index' of the radio button is now controlled by our session state variable.
selected_tab_title = st.sidebar.radio(
    "Go to", TABS,
    key="navigation_radio",
    index=st.session_state.active_tab_index
)

# Keep a separate variable for the title for clarity in the if/elif blocks
selected_tab = selected_tab_title

# --- PERFORMANCE DASHBOARD TAB ---
if selected_tab == "📈 Dashboard":
    st.header("📈 Performance Dashboard")
    st.markdown("An at-a-glance overview of AI performance and user interaction.")

    try:
        # --- Define the callback functions ONCE at the top level ---
        def set_question_to_curate(q):
            # 1. Set the question text to be curated
            st.session_state.question_to_curate = q

            # 2. Find the index of the "Curated Q&A" tab
            qna_tab_index = TABS.index("📝 Curated Q&A")

            # 3. Programmatically change the active tab index in session state
            st.session_state.active_tab_index = qna_tab_index

            # The toast message is now less critical but still nice to have
            st.toast("Switching to the Curated Q&A tab...", icon="📝")
            # We don't need to rerun; the state change will handle it on the next cycle.
            
        # New callback for resetting counts
        def handle_reset_counts(q):
            with st.spinner(f"Resetting all historical logs for question: '{q[:50]}...'"):
                feedback_deleted, query_deleted = reset_logs_for_question(q)
                st.success(f"Reset complete! Removed {feedback_deleted} feedback and {query_deleted} query logs.")

        # --- 1. Key Performance Indicators (KPIs) ---
        kpis = get_dashboard_kpis()
        st.subheader("Key Metrics")
        kpi_cols = st.columns(3)
        # ... (KPI code remains the same) ...
        kpi_cols[0].metric("Total Queries", f"{kpis['total_queries']:,}")
        kpi_cols[1].metric("Total Feedbacks", f"{kpis['total_feedback']:,}")
        feedback_ratio_str = f"{kpis['positive_feedback']} 👍 / {kpis['negative_feedback']} 👎"
        kpi_cols[2].metric("Feedback Ratio (Up/Down)", feedback_ratio_str)
        
        st.divider()

        # --- 2. Top Problem Areas with new Reset button ---
        problem_cols = st.columns(2)
        with problem_cols[0]:
            st.subheader("Top Problem Questions (👎)")
            problem_df = get_top_problem_questions()
            if not problem_df.empty:
                header_cols = st.columns([1, 3, 1, 1]) # Adjusted column ratios
                header_cols[0].markdown("**Downvotes**")
                header_cols[1].markdown("**Question**")
                header_cols[2].markdown("**Curate**")
                header_cols[3].markdown("**Reset**")

                for index, row in problem_df.iterrows():
                    row_cols = st.columns([1, 3, 1, 1]) # Adjusted column ratios
                    row_cols[0].text(row['downvote_count'])
                    row_cols[1].text(row['question'])
                    row_cols[2].button("✍️", key=f"prob_curate_{index}", on_click=set_question_to_curate, args=(row['question'],), help="Copy question to Curated Q&A tab")
                    row_cols[3].button("🔄", key=f"prob_reset_{index}", on_click=handle_reset_counts, args=(row['question'],), help="Reset historical counts for this question")
            else:
                st.info("No downvoted questions yet. Good job!")

        with problem_cols[1]:
            st.subheader("Top Information Gaps")
            gaps_df = get_top_information_gaps()
            if not gaps_df.empty:
                header_cols = st.columns([1, 3, 1, 1]) # Adjusted column ratios
                header_cols[0].markdown("**Failures**")
                header_cols[1].markdown("**Question**")
                header_cols[2].markdown("**Curate**")
                header_cols[3].markdown("**Reset**")
                
                for index, row in gaps_df.iterrows():
                    row_cols = st.columns([1, 3, 1, 1]) # Adjusted column ratios
                    row_cols[0].text(row['failure_count'])
                    row_cols[1].text(row['question'])
                    row_cols[2].button("✍️", key=f"gap_curate_{index}", on_click=set_question_to_curate, args=(row['question'],), help="Copy question to Curated Q&A tab")
                    row_cols[3].button("🔄", key=f"gap_reset_{index}", on_click=handle_reset_counts, args=(row['question'],), help="Reset historical counts for this question")
            else:
                st.info("No information gaps detected yet.")
            
        st.divider()

        # --- 3. Most Frequent Questions Chart ---
        # ... (This section remains the same) ...
        st.subheader("Most Frequent User Questions")
        freq_df = get_most_frequent_questions()
        if not freq_df.empty:
            freq_df.set_index('question', inplace=True)
            st.bar_chart(freq_df)
        else:
            st.info("No questions have been logged yet.")

    except Exception as e:
        st.error(f"Could not load dashboard data. Error: {e}")


# --- AI CONFIGURATION TAB ---
elif selected_tab == "🤖 AI Config":
    st.header("AI System Configuration")
    st.markdown("Adjust the core behavior of the AI. Changes will take effect immediately for all users.")

    try:
        # Load the current configuration from the database
        current_config = get_ai_config()

    # Display the currently used models for clarity
        st.info(f"""
        **Currently Active Models:**
        - **Language Model (LLM):** `{LLM_MODEL}` (This is set in the backend code)
        - **Embedding Model:** `{current_config['embedding_model']}` (Configurable below)
        """)
        st.divider()

        with st.form("ai_config_form"):
            st.subheader("Master System Prompt")
            st.markdown("This is the core instruction given to the AI. It defines its personality, rules, and response format.")
            system_prompt = st.text_area(
                "System Prompt",
                value=current_config['system_prompt'],
                height=250,
                key="ai_config_system_prompt" # Unique key for this widget
            )

            st.subheader("Retriever Settings")
            st.markdown("Control how the AI finds relevant documents.")
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.number_input("Top-K", min_value=1, max_value=20, value=int(current_config['top_k']), help="The maximum number of documents to retrieve.", key="ai_config_top_k")
            with col2:
                score_threshold = st.slider("Score Threshold", min_value=0.0, max_value=1.0, value=float(current_config['score_threshold']), help="The minimum relevance score for a document to be considered. Higher is stricter.", key="ai_config_score_threshold")

            st.subheader("Embedding Model")
            st.markdown("The model used to understand the meaning of text for searching. **Changing this requires re-ingesting all documents.**")
            embedding_model = st.text_input("Embedding Model Name", value=current_config['embedding_model'], help="e.g., all-MiniLM-L6-v2, bge-large-en-v1.5", key="ai_config_embedding_model")

            # The submit button is inside the form and has NO 'key' argument.
            submitted = st.form_submit_button("Save Configuration")

            if submitted:
                if embedding_model != current_config['embedding_model']:
                    st.warning("IMPORTANT: Embedding model changed. You MUST delete and re-upload all documents for changes to take effect.")
                with st.spinner("Saving configuration..."):
                    update_ai_config(system_prompt, score_threshold, top_k, embedding_model)
                    st.cache_resource.clear()
                    st.success("Configuration saved! The AI has been updated.")
    
    except Exception as e:
        st.error(f"Could not load AI configuration. Error: {e}")

# --- CHAT INTERFACE ---
elif selected_tab == "💬 Chat with AI":
    st.header("Ask a Question")
    try:
        retriever, generation_chain = get_chains()
        ctx = get_script_run_ctx()
        session_id = ctx.session_id if ctx else "local_session"
    except Exception as e:
        st.error(f"Failed to initialize RAG chain: {e}")
        st.stop()

    def handle_feedback(message_index, rating):
        message = st.session_state.messages[message_index]
        log_feedback(session_id, message.get("question_asked", "N/A"), message["content"], message.get("sources", []), rating)
        st.session_state.messages[message_index]["feedback_given"] = True
        st.toast("Thank you for your feedback!", icon="✅")

    if "messages" not in st.session_state: st.session_state.messages = []
    
    # --- CHANGE 1: In the history display loop ---
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if "sources" in message:
                    with st.expander("View Sources for this response"):
                        for source in message["sources"]:
                            source_name = source.metadata.get('source_doc_name', 'N/A')
                            if source_name.startswith("http"):
                                link_text = f"**Source:** [{source_name}]({source_name})"
                            else:
                                link_text = f"**Source:** {source_name}"
                            st.markdown(link_text)
                            st.info(f"**Content:** {source.page_content}")

                if not message.get("feedback_given", False):
                    cols = st.columns([1, 1, 10])
                    cols[0].button("👍", key=f"up_{i}", on_click=handle_feedback, args=(i, 1))
                    cols[1].button("👎", key=f"down_{i}", on_click=handle_feedback, args=(i, -1))

    if prompt := st.chat_input("How can I help you today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Finding documents..."):
                source_docs = retriever.invoke(prompt)
            with st.spinner("Generating answer..."):
                context_text = "\n\n".join([doc.page_content for doc in source_docs])
                chain_input = {"context": context_text, "question": prompt}
                answer = st.write_stream(generation_chain.stream(chain_input))
            
            log_query(session_id, prompt, context_text, answer)
            
            response_to_history = {"role": "assistant", "content": answer, "question_asked": prompt}
            if "enough information" not in answer.lower():
                response_to_history["sources"] = source_docs
                
                # --- CHANGE 2: In the new response display ---
                with st.expander("View Sources for this response"):
                    for source in source_docs:
                        source_name = source.metadata.get('source_doc_name', 'N/A')
                        if source_name.startswith("http"):
                            link_text = f"**Source:** [{source_name}]({source_name})"
                        else:
                            link_text = f"**Source:** {source_name}"
                        st.markdown(link_text)
                        st.info(f"**Content:** {source.page_content}")

            st.session_state.messages.append(response_to_history)
            st.rerun()

# --- DOCUMENT MANAGEMENT INTERFACE ---
elif selected_tab == "🗂️ Manage Documents":
    st.header("Upload and Manage Your Knowledge Base")
    
    st.subheader("Bulk Ingest from a Directory")
    st.markdown("Provide the full path to a directory on the server. The app will recursively scan all subfolders and ingest any supported documents it finds.")

    dir_to_scan = st.text_input(
        "Enter the full path to the directory to scan",
        placeholder="/home/user/product_docs_main_folder"
    )

    if st.button("Start Directory Scan and Ingest"):
        if dir_to_scan and os.path.isdir(dir_to_scan):
            with st.spinner(f"Scanning directory {dir_to_scan}... This may take a while."):
                try:
                    # The backend function now returns two lists
                    processed_files, failed_files = add_directory_to_db(dir_to_scan)
                    
                    # Display the results in a user-friendly way
                    st.success(f"Scan complete! Processed {len(processed_files)} files successfully.")
                    if processed_files:
                        with st.expander("Show successfully processed files"):
                            st.write(processed_files)
                    
                    if failed_files:
                        st.error(f"Failed to process {len(failed_files)} files.")
                        with st.expander("Show failed files and errors"):
                            st.write(failed_files)

                    # Clear cache once after the entire batch is done
                    st.cache_resource.clear()
                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred during the directory scan: {e}")
        elif not dir_to_scan:
            st.warning("Please enter a directory path.")
        else:
            st.error(f"The path provided does not exist or is not a directory: {dir_to_scan}")

    st.divider() # Add a divider to separate this from the other upload methods

      # --- Section 2: File Uploader (Corrected with st.form) ---
    st.subheader("Upload Individual Files")

    # Wrap the uploader and button in a form
    with st.form("file_upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "docx", "xlsx", "pptx", "csv", "md"],
            accept_multiple_files=True,
            key="file_uploader_widget"
        )

        # The submission button for the form
        submitted = st.form_submit_button("Add Uploaded Documents")

        if submitted:
            # The logic runs only when the form is submitted
            if not uploaded_files:
                st.warning("Please select at least one document to upload.")
            else:
                for uploaded_file in uploaded_files:
                    temp_dir = "temp_data"
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)
                    
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        try:
                            add_document_to_db(file_path, file_type)
                            st.success(f"Processed {uploaded_file.name}!")
                        except Exception as e:
                            st.error(f"Failed to process {uploaded_file.name}: {e}")
                        finally:
                            os.remove(file_path)
                
                st.cache_resource.clear()
                st.info("All documents processed. The list of existing documents will be updated.")
                # We don't need st.rerun() here, as form submission handles the rerun.


    # --- Section 2: URL Ingestion (The code you already had) ---
    st.divider()
    st.subheader("Add Content from a Web Page")

    def handle_add_url():
        url = st.session_state.url_input_for_callback
        if url:
            with st.spinner(f"Fetching and processing content from {url}..."):
                try:
                    add_url_to_db(url)
                    st.cache_resource.clear()
                    st.success(f"Content from URL added successfully!")
                    st.session_state.url_input_for_callback = "" 
                except Exception as e:
                    st.error(f"Failed to add content from URL: {e}")
        else:
            st.warning("Please enter a URL.")

    st.text_input("Enter a URL to ingest", key="url_input_for_callback")
    st.button("Add URL to Knowledge Base", on_click=handle_add_url)

    # --- Section 3: Slack Ingestion (The code you already had) ---
    st.divider()
    st.subheader("Add Content from a Slack Export")
    # You might need to adjust this path or import it from your backend
    SLACK_IMPORT_BASE_PATH = "/home/kburr/slack_imports" 
    st.markdown(f"1. Unzip your Slack export into a new subfolder inside: `{SLACK_IMPORT_BASE_PATH}`")
    st.markdown("2. Enter the name of that new subfolder below.")
    export_folder_name = st.text_input("Enter the name of the unzipped Slack export subfolder", placeholder="e.g., my-team-export-feb-2025")
    if st.button("Add Slack Export to Knowledge Base"):
        if export_folder_name:
            with st.spinner(f"Processing Slack export from {export_folder_name}..."):
                try:
                    add_slack_dir_to_db(export_folder_name)
                    st.cache_resource.clear()
                    st.success("Content from Slack export added successfully!")
                except Exception as e:
                    st.error(f"Failed to add content from Slack export: {e}")
        else:
            st.warning("Please enter the name of the export subfolder.")

    # --- Section 4: The Existing Documents List (The code you already had) ---
    st.divider()
    st.subheader("Existing Documents in Knowledge Base")
    try:
        all_docs = list_all_documents()
        if not all_docs:
            st.info("No documents have been added yet.")
        else:
            for doc in all_docs:
                col1, col2 = st.columns([4, 1])
                col1.text(doc["name"])
                if col2.button("Delete", key=f"delete_{doc['id']}"):
                    with st.spinner(f"Deleting {doc['name']}..."):
                        delete_document_by_id(doc['id'])
                        st.cache_resource.clear()
                        st.success(f"Deleted {doc['name']}.")
                        st.rerun()
    except Exception as e:
        st.error(f"Could not connect to the database to list documents. Is it running? Error: {e}")
        
     # --- Section 6: Danger Zone ---
    st.divider()
    st.subheader("🚨 Danger Zone")
    
    # Use a checkbox as a safety measure
    confirm_delete = st.checkbox("I understand this is irreversible and I want to delete ALL documents and data.")

    if confirm_delete:
        if st.button("DELETE ALL KNOWLEDGE BASE DATA"):
            with st.spinner("Deleting all data from the knowledge base... This cannot be undone."):
                success = delete_all_data()
                if success:
                    st.cache_resource.clear() # CRITICAL: Clear all app cache
                    st.success("All knowledge base data has been permanently deleted.")
                    # A small delay before rerunning can improve user experience 
                    st.rerun()
                else:
                    st.error("An error occurred while deleting data. Check the terminal for logs.")


# --- CURATED Q&A MANAGEMENT INTERFACE ---

elif selected_tab == "📝 Curated Q&A":
    st.header("Manage Curated Q&A")
    st.markdown("Add, review, and delete 'golden' question-and-answer pairs. These will be strongly preferred by the AI.")

    # --- Section to add a new Q&A ---
    with st.form("new_qa_form", clear_on_submit=True):
        st.subheader("Add a New Q&A Pair")
        
        # --- START OF NEW CODE ---
        # If a question was sent from the dashboard, use it as the default value.
        # Otherwise, the default is an empty string.
        default_question = st.session_state.get("question_to_curate", "")
        new_question = st.text_input(
            "Question", 
            value=default_question, 
            placeholder="e.g., How do I reset my password?",
            key="qna_form_question"
        )
        new_answer = st.text_area("Answer", 
                                  placeholder="e.g., To reset your password, go to Settings > Account > Reset Password."
                                  )
        submitted = st.form_submit_button("Add Q&A")

        if submitted:
            if new_question and new_answer:
                with st.spinner("Adding Q&A..."):
                    add_curated_qa(new_question, new_answer)
                    st.cache_resource.clear() # IMPORTANT: Clear cache
                    st.success("Q&A pair added successfully!")
            else:
                st.error("Please provide both a question and an answer.")

    st.divider()

    # --- Section to manage existing Q&A ---
    st.subheader("Existing Q&A Pairs")
    
    try:
        all_qa = get_all_curated_qa()
        if all_qa.empty:
            st.info("No curated Q&A pairs have been added yet.")
        else:
            # Display current Q&A and a delete button for each
            for index, row in all_qa.iterrows():
                # --- START OF NEW CODE ---
                # Display the feedback counts using markdown and emojis
                st.markdown(f"**Feedback:** 👍 {row['thumbs_up']}    👎 {row['thumbs_down']}")
                # --- END OF NEW CODE ---
                
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.text_input("Question", value=row["question"], disabled=True, key=f"q_{row['id']}")
                    st.text_area("Answer", value=row["answer"], disabled=True, key=f"a_{row['id']}")
                with col2:
                    if st.button("Delete", key=f"del_{row['id']}"):
                        with st.spinner("Deleting Q&A..."):
                            delete_curated_qa(str(row["id"]))
                            st.cache_resource.clear() # IMPORTANT: Clear cache
                            st.success("Q&A pair deleted.")
                            st.rerun() # Refresh the page to show the updated list
                st.markdown("---")

    except Exception as e:
        st.error(f"Could not load curated Q&A. Error: {e}")
        
# --- FEEDBACK LOG REVIEW INTERFACE ---
elif selected_tab == "📊 Feedback Log":
    st.header("Review User Feedback")
    st.markdown("Analyze feedback to identify gaps in documentation and improve AI performance.")

    try:
        feedback_df = get_feedback_log()
        
        if feedback_df.empty:
            st.info("No feedback has been submitted yet.")
        else:
            # Add a filter for thumbs up/down
            feedback_filter = st.radio(
                "Filter by rating:",
                ("All", "👍 Thumbs Up", "👎 Thumbs Down"),
                horizontal=True
            )

            # Apply the filter
            if feedback_filter == "👍 Thumbs Up":
                filtered_df = feedback_df[feedback_df['rating'] == 1]
            elif feedback_filter == "👎 Thumbs Down":
                filtered_df = feedback_df[feedback_df['rating'] == -1]
            else:
                filtered_df = feedback_df

            if filtered_df.empty:
                st.warning("No feedback matches the selected filter.")
            else:
                # Display each piece of feedback as a card
                for index, row in filtered_df.iterrows():
                    rating_emoji = "👍" if row['rating'] == 1 else "👎"
                    
                    with st.container(border=True):
                        st.markdown(f"**{rating_emoji} Feedback received on:** {row['created_at'].strftime('%Y-%m-%d %H:%M')}")
                        st.text_input("Question", value=row['question'], disabled=True, key=f"fb_q_{index}")
                        st.text_area("Answer Provided", value=row['answer'], disabled=True, key=f"fb_a_{index}")
                        
                        # Show the sources that were used for this answer
                        if row['sources']:
                            with st.expander("View sources used for this answer"):
                                # The 'sources' column is a JSON string, so we need to parse it
                                sources_data = row['sources']
                                if not sources_data:
                                    st.write("No sources were recorded for this answer.")
                                else:
                                    for source in sources_data:
                                        st.info(f"**Document:** {source['metadata'].get('source_doc_name', 'N/A')}\n\n**Content:** {source['page_content']}")
                        else:
                             st.info("No sources were recorded for this answer.")

    except Exception as e:
        st.error(f"Could not load feedback log. Error: {e}")
        
# --- QUERY INSPECTOR INTERFACE ---
elif selected_tab == "🔍 Query Inspector":
    st.header("Query Inspector")
    st.markdown("Search, filter, and export the end-to-end flow of each query.")

    # --- Search and Export Expander ---
    with st.expander("🔍 Search and Export Options", expanded=True):
        # Create columns for a cleaner layout
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", None)
        with col2:
            end_date = st.date_input("End date", None)

        search_term = st.text_input("Search in question or answer", placeholder="e.g., password, Quantum Sync")
        
        # Fetch the filtered data from the backend
        try:
            query_df = get_query_log(start_date, end_date, search_term)
        except Exception as e:
            st.error(f"Could not load query log. Error: {e}")
            query_df = pd.DataFrame() # Create an empty df on error

        # --- Export Button ---
        if not query_df.empty:
            csv_data = query_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Results to CSV",
                data=csv_data,
                file_name=f"query_log_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.warning("No data to export for the current filter.")
    
    st.divider()

    # --- Results Display ---
    st.subheader(f"Displaying {len(query_df)} Results")
    
    if query_df.empty:
        st.info("No queries match your current filter criteria.")
    else:
        # The compressed view toggle and display logic remains the same
        compressed_view = st.toggle("Enable Compressed Log View", value=True)
        
        if compressed_view:
            for index, row in query_df.iterrows():
                expander_title = f"🗓️ {row['created_at'].strftime('%Y-%m-%d %H:%M')} - {row['question']}"
                with st.expander(expander_title):
                    st.markdown(f"**1. Retrieved Context from Database:**")
                    st.code(row['retrieved_context'], language=None)
                    st.markdown(f"**2. Final Answer from LLM:**")
                    st.markdown(f"> {row['final_answer']}")
        else: # Detailed View
            for index, row in query_df.iterrows():
                with st.container(border=True):
                    st.markdown(f"**🗓️ Logged on:** {row['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.subheader("1. User's Question")
                    st.text_area("Question", value=row['question'], disabled=True, key=f"q_log_{index}")
                    st.subheader("2. Retrieved Context from Database")
                    st.text_area("Context", value=row['retrieved_context'], height=200, disabled=True, key=f"c_log_{index}")
                    st.subheader("3. Final Answer from LLM")
                    st.text_area("Answer", value=row['final_answer'], height=100, disabled=True, key=f"a_log_{index}")