import os
import streamlit as st
from dotenv import load_dotenv
from core import build_index, retrieve, answer

load_dotenv()
st.set_page_config(page_title='Document Q&A', page_icon='📚', layout='wide')
st.title('📚 Intelligent Document Q&A')
st.caption('Upload documents, ask questions, and inspect the evidence behind each answer.')
with st.sidebar:
    st.header('Documents & retrieval')
    files = st.file_uploader('PDF, DOCX, TXT or Markdown', type=['pdf', 'docx', 'txt', 'md'], accept_multiple_files=True)
    size = st.select_slider('Chunk size (tokens)', options=[256, 384, 512, 768], value=384)
    overlap = st.slider('Chunk overlap (tokens)', 0, 128, 64)
    k = st.slider('Retrieved chunks', 1, 10, 4)
    distance = st.slider('Maximum cosine distance', 0.0, 2.0, 0.9, 0.05)
    evidence_only = st.checkbox('Retrieval only (no LLM needed)', value=True)
    st.caption('Lower distance is a closer match. This is not a probability of correctness.')
    st.caption('Provider: ' + os.getenv('LLM_PROVIDER', 'ollama'))
    if st.button('Build index', type='primary'):
        st.session_state.pop('store', None)
        st.session_state.pop('indexed_signature', None)
        try:
            if sum(f.size for f in files) > 30 * 1024 * 1024:
                raise ValueError('Please keep total upload size below 30 MB.')
            with st.spinner('Extracting text, creating embeddings and indexing…'):
                payload = [(f.name, f.getvalue()) for f in files]
                store, count = build_index(payload, size, overlap)
                from core import corpus_id
                st.session_state.store = store
                st.session_state.indexed_signature = corpus_id(payload, size, overlap)
            st.success(f'Indexed {len(files)} files into {count} chunks.')
        except Exception as exc:
            st.error(f'Indexing failed: {exc}')
question = st.text_input('Ask a question', placeholder='What is the refund policy?')
if st.button('Find answer'):
    from core import corpus_id
    current = corpus_id([(f.name, f.getvalue()) for f in files], size, overlap)
    if 'store' not in st.session_state or current != st.session_state.get('indexed_signature'):
        st.warning('Build the index for the currently selected files and chunk settings first.')
    elif not question.strip():
        st.warning('Enter a question first.')
    else:
        try:
            with st.spinner('Retrieving evidence…'):
                hits = retrieve(st.session_state.store, question, k, distance)
                if not evidence_only:
                    result, status = answer(question, hits)
                    st.markdown(result)
                    st.caption(f'Citation ID check: {status}. Valid IDs do not prove that a claim is supported.')
                elif not hits:
                    st.info('No evidence passed the retrieval threshold.')
            st.subheader('Retrieved evidence')
            for i, (doc, score) in enumerate(hits, 1):
                with st.expander(f'[{i}] {doc.metadata["source"]} · page/section {doc.metadata["page"]} · distance {score:.3f}', expanded=True):
                    st.write(doc.page_content)
            st.download_button('Download evidence', '\n\n'.join(d.page_content for d, _ in hits), 'evidence.txt')
        except Exception as exc:
            st.error(f'Query failed: {exc}')
