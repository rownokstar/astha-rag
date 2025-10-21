import streamlit as st
import os
import tempfile
import warnings
from huggingface_hub import hf_hub_download

# LangChain components
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Suppress warnings
warnings.filterwarnings('ignore')

# --- অ্যাপের টাইটেল এবং কনফিগারেশন ---
st.set_page_config(page_title="Local RAG Chatbot", page_icon="🤖")
st.title("📄🤖 অ্যাডভান্সড RAG চ্যাটবট")
st.write("আপনার PDF আপলোড করুন (যাতে টেক্সট, টেবিল এবং ছবি থাকতে পারে) এবং প্রশ্ন করুন।")

# --- ক্যাশড ফাংশন (যাতে রিসোর্স বারবার লোড না হয়) ---

@st.cache_resource
def load_llm():
    """
    লোকাল GGUF LLM (Phi-3-mini) লোড করে।
    """
    try:
        st.write("Downloading local LLM (Phi-3-mini GGUF)... This may take a few minutes on first run.")
        model_name_or_path = "microsoft/Phi-3-mini-4k-instruct-gguf"
        model_file = "Phi-3-mini-4k-instruct-Q4_K_M.gguf"
        
        model_path = hf_hub_download(
            repo_id=model_name_or_path,
            filename=model_file
        )
        
        st.write("Loading LLM into memory...")
        # নোট: লোকাল মেশিনে GPU না থাকলে 'gpu_layers' 0 রাখুন।
        # আপনার NVIDIA GPU থাকলে এটি বাড়াতে পারেন (যেমন: 20-50)।
        llm = CTransformers(
            model=model_path,
            model_type='llama',
            config={
                'context_length': 4096,
                'gpu_layers': 0, # লোকাল CPU-এর জন্য 0; GPU থাকলে বাড়ান
                'temperature': 0.1
            }
        )
        st.write("Local LLM loaded successfully.")
        return llm
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        return None

@st.cache_resource
def create_rag_pipeline(uploaded_file, _embeddings):
    """
    আপলোড করা PDF থেকে RAG পাইপলাইন (ভেক্টর স্টোর এবং রিট্রিভার) তৈরি করে।
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        st.write(f"Processing PDF '{uploaded_file.name}' with OCR and table extraction...")
        # UnstructuredPDFLoader ব্যবহার করে সব ধরনের কন্টেন্ট এক্সট্রাক্ট করা
        loader = UnstructuredPDFLoader(
            tmp_file_path, 
            strategy="hi_res", 
            mode="elements"
        )
        docs = loader.load()
        os.remove(tmp_file_path) # টেম্পোরারি ফাইল ডিলিট করুন

        if not docs:
            st.error("Could not extract any content from the PDF.")
            return None

        st.write(f"Extracted {len(docs)} elements. Now creating vector store...")
        
        # Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splitted_docs = text_splitter.split_documents(docs)
        
        # FAISS Vector Store (লোকাল) - এখানে faiss-cpu ব্যবহৃত হচ্ছে
        vector_store = FAISS.from_documents(splitted_docs, _embeddings)
        st.write("Vector store created successfully.")
        
        return vector_store
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

@st.cache_resource
def load_embedding_model():
    """
    এমবেডিং মডেল লোড করে।
    """
    st.write("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# --- কাস্টম প্রম্পট টেমপ্লেট ---
prompt_template = """<|system|>
You are a helpful assistant. Answer the user's question based only on the following context:
{context}
<|end|>
<|user|>
{question}
<|end|>
<|assistant|>"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# --- Streamlit অ্যাপের প্রধান ইন্টারফেস ---

# ১. রিসোর্স লোড করুন
embeddings = load_embedding_model()
llm = load_llm()

# ২. চ্যাট সেশন স্টেট ইনিশিয়ালাইজ করুন
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ৩. ফাইল আপলোডার
with st.sidebar:
    st.header("১. PDF আপলোড করুন")
    uploaded_file = st.file_uploader("আপনার PDF ফাইলটি এখানে আপলোড করুন", type="pdf")
    
    if uploaded_file:
        if st.button("প্রসেস শুরু করুন"):
            with st.spinner("PDF প্রসেস করা হচ্ছে... (OCR এবং টেবিল সহ) এটাতে সময় লাগতে পারে।"):
                # RAG পাইপলাইন তৈরি করুন
                vector_store = create_rag_pipeline(uploaded_file, embeddings)
                
                if vector_store and llm:
                    # RAG চেইন তৈরি করুন এবং সেশনে সেভ করুন
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                        chain_type_kwargs={"prompt": PROMPT},
                        return_source_documents=True
                    )
                    st.session_state.messages = [] # নতুন ফাইল আপলোড হলে চ্যাট রিসেট করুন
                    st.success(f"'{uploaded_file.name}' সফলভাবে প্রসেস করা হয়েছে। এখন আপনি প্রশ্ন করতে পারেন।")
                else:
                    st.error("RAG পাইপলাইন তৈরি করা সম্ভব হয়নি।")

# ৪. চ্যাট ইন্টারফেস
st.header("২. চ্যাট করুন")

if not st.session_state.qa_chain:
    st.warning("অনুগ্রহ করে সাইডবারে একটি PDF ফাইল আপলোড করুন এবং 'প্রসেস শুরু করুন' বাটনে ক্লিক করুন।")

# আগের চ্যাট মেসেজগুলো দেখান
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ইউজারের কাছ থেকে নতুন ইনপুট নিন
if prompt := st.chat_input("আপনার প্রশ্ন লিখুন...", disabled=(not st.session_state.qa_chain)):
    # ইউজার মেসেজ দেখান
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI-এর উত্তর জেনারেট করুন
    with st.chat_message("assistant"):
        with st.spinner("উত্তর খোঁজা হচ্ছে..."):
            try:
                result = st.session_state.qa_chain.invoke(prompt)
                response = result['result']
                
                # সোর্স ডকুমেন্ট দেখানো (ঐচ্ছিক)
                if result.get('source_documents'):
                    with st.expander("সোর্স দেখুন"):
                        for i, doc in enumerate(result['source_documents']):
                            page = doc.metadata.get('page_number', 'N/A')
                            st.write(f"**সোর্স {i+1} (Page: {page})**")
                            st.caption(f"{doc.page_content[:200]}...")
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"An error occurred while generating response: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
