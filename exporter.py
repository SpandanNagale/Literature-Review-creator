import streamlit as st


def export_markdown(content: str, filename="output.md"):
    if not content:
        st.warning("No content to export.")
        return
    
    st.download_button(
        label="📥 Download Review as Markdown",
        data=content.encode("utf-8"),
        file_name=filename,
        mime="text/markdown"
    )
