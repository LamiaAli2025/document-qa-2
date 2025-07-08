import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
import numpy as np
import faiss
from typing import List, Dict
import requests
from io import BytesIO
import torch
from transformers import AutoTokenizer, AutoModel

# Disable TensorFlow completely
os.environ['NO_TF'] = '1'

# PyTorch-based embedding model
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def generate_embeddings(texts, tokenizer, model):
    encoded_input = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        return_tensors='pt',
        max_length=128
    )
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Use mean pooling with attention masking
    token_embeddings = model_output.last_hidden_state
    input_mask = encoded_input['attention_mask']
    token_embeddings = token_embeddings.masked_fill(~input_mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / input_mask.sum(dim=1)[..., None]
    
    return sentence_embeddings.numpy()

# OCR RAG System class definition
class OCR_RAG_System:
    def __init__(self, 
                 tokenizer,
                 model,
                 knowledge_base_path: str = None):
        self.tokenizer = tokenizer
        self.model = model
        self.embedding_dim = 768  # Fixed dimension for this model
        self.knowledge_base_path = knowledge_base_path
        
        # Initialize Gemini
        genai.configure(api_key="AIzaSyAlrNvpn6unHgyYB1pDSouI_bB-0pB_05w")
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize knowledge base
        self.knowledge_base = {}
        self.index = None
        if knowledge_base_path and os.path.exists(knowledge_base_path):
            self.load_knowledge_base()
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)

    def extract_text_from_image(self, img) -> str:
        """Extract text using Gemini Vision"""
        try:
            response = self.gemini_model.generate_content([
                "EXTRACT ARABIC TEXT EXACTLY AS WRITTEN. NO TRANSLATIONS. NO COMMENTS. PRESERVE SPACING/LINE BREAKS:",
                img
            ])
            return response.text.strip()
        except Exception as e:
            return ""

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using PyTorch model"""
        if not text.strip():
            return np.zeros(self.embedding_dim)
        return generate_embeddings([text], self.tokenizer, self.model)[0]

    def add_to_knowledge_base(self, text: str, metadata: Dict = None) -> None:
        """Add text to knowledge base"""
        embedding = self.generate_embedding(text)
        doc_id = str(len(self.knowledge_base))
        
        self.knowledge_base[doc_id] = {
            'text': text,
            'embedding': embedding,
            'metadata': metadata or {}
        }
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(np.array([embedding]))

    def retrieve_similar_documents(self, query_text: str, k: int = 3) -> List[Dict]:
        """Find similar documents in knowledge base"""
        query_embedding = self.generate_embedding(query_text)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.knowledge_base) and distance <= 0.4:
                doc_id = str(idx)
                doc = self.knowledge_base[doc_id]
                results.append({
                    'text': doc['text'],
                    'distance': float(distance)
                })
        return results

    def enhance_ocr_output(self, extracted_text: str, img, context: List[Dict]) -> str:
        """Improve OCR results using context"""
        if not context:
            return extracted_text
            
        context_texts = [doc['text'] for doc in context]
        prompt = f"""
        CORRECT THE ARABIC TEXT EXTRACTED FROM AN IMAGE USING THE CONTEXT BELOW:
        
        Extracted Text: 
        {extracted_text}
        
        Context (Similar Texts):
        {''.join(context_texts)}
        
        Rules:
        1. ONLY CORRECT OBVIOUS ERRORS USING CONTEXT
        2. PRESERVE ORIGINAL FORMATTING, SPACING AND LINE BREAKS
        3. DO NOT ADD NEW CONTENT
        4. RETURN ONLY THE CORRECTED ARABIC TEXT
        """
        
        try:
            response = self.gemini_model.generate_content([prompt, img])
            return response.text.strip()
        except:
            return extracted_text

    def process_image(self, img, use_enhancement: bool) -> str:
        """Full OCR processing pipeline"""
        extracted_text = self.extract_text_from_image(img)
        
        # Add to knowledge base even if empty
        self.add_to_knowledge_base(extracted_text, {'source': 'upload'})
        
        if not use_enhancement:
            return extracted_text or "No text recognized"
            
        similar_docs = self.retrieve_similar_documents(extracted_text)
        if similar_docs:
            return self.enhance_ocr_output(extracted_text, img, similar_docs)
        return extracted_text

    def save_knowledge_base(self):
        """Persist knowledge base to disk"""
        import pickle
        if self.knowledge_base_path:
            with open(self.knowledge_base_path, 'wb') as f:
                pickle.dump({
                    'knowledge_base': self.knowledge_base,
                    'index': self.index
                }, f)

    def load_knowledge_base(self):
        """Load knowledge base from disk"""
        import pickle
        if self.knowledge_base_path and os.path.exists(self.knowledge_base_path):
            with open(self.knowledge_base_path, 'rb') as f:
                data = pickle.load(f)
                self.knowledge_base = data['knowledge_base']
                self.index = data['index']

# UI Configuration
st.set_page_config(page_title="Arabic Handwriting Extraction", page_icon="??")
st.title("?? Arabic Handwriting Extraction")

# Model selection
model_choice = st.radio("Select recognition model:", 
                        ["Google Gemini OCR", "RAG Arabic Handwriting with Gemini OCR Model", "Learnned Model"])

# Image uploader
uploaded_file = st.file_uploader(
    "Upload Handwriting Image",
    type=["png", "jpg", "jpeg"],
    help="Take a clear photo of handwritten Arabic text"
)

if uploaded_file:
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Extract Text"):
        if model_choice == "Google Gemini OCR":
            # Gemini processing
            genai.configure(api_key="AIzaSyAlrNvpn6unHgyYB1pDSouI_bB-0pB_05w")
            
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                prompt = "Extract Arabic handwriting EXACTLY as written. Return only raw text."
                
                with st.spinner("Extracting with Gemini..."):
                    response = model.generate_content(
                        [prompt, {"mime_type": uploaded_file.type, "data": uploaded_file.getvalue()}]
                    )
                
                st.subheader("Extracted Text (Gemini)")
                st.write(response.text)
                
            except Exception as e:
                st.error(f"Gemini Error: {str(e)}")
        
        elif model_choice == "RAG Arabic Handwriting with Gemini OCR Model":
            try:
                with st.spinner("Loading model and extracting text..."):
                    # Load PyTorch model
                    tokenizer, model = load_embedding_model()
                    
                    # Initialize OCR system
                    ocr_system = OCR_RAG_System(
                        tokenizer=tokenizer,
                        model=model,
                        knowledge_base_path='knowledge_base.pkl'
                    )
                    
                    # Process image
                    result = ocr_system.process_image(pil_image, True)
                    
                    # Save knowledge base
                    ocr_system.save_knowledge_base()
                    
                    # Display results
                    st.subheader("Extracted Text (RAG Model)")
                    st.write(result or "No text recognized")
        
            except Exception as e:
                st.error(f"Processing Error: {str(e)}")
        elif model_choice == "Learnned Model":
            try:        
                st.subheader("Extracted Text (Learnned Model) ...... comming soon")
                
            except Exception as e:
                st.error(f"Processing Error: {str(e)}")    