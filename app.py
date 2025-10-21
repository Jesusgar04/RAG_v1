"""
CHATBOT SECRETARÍA ACADÉMICA - UNIVERSIDAD LIBRE
Sistema RAG para consultas académicas
"""

import streamlit as st
import os
import json
from pathlib import Path
import time
import base64

# Importaciones de LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# ==================== CONFIGURACIÓN DE LA PÁGINA ====================

st.set_page_config(
    page_title="Chatbot Secretaría Académica | Universidad Libre",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==================== ESTILOS CSS LIMPIOS ====================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Ocultar elementos */
    #MainMenu, footer, header, .stDeployButton { 
        visibility: hidden; 
    }

    /* Fondo limpio */
    .stApp {
        background: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        max-width: 850px;
        padding: 1rem 1.5rem 3rem 1.5rem;
    }

    /* ========== HEADER MINIMALISTA ========== */
    .header-minimal {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.5rem;
        background: red;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }

    .header-minimal img {
        width: 50px;
        height: 50px;
        border-radius: 8px;
    }

    .header-text h1 {
        margin: 0;
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a1a;
    }

    .header-text p {
        margin: 0;
        font-size: 0.85rem;
        color: #6c757d;
    }

    /* ========== MENSAJES DEL CHAT ========== */
    .stChatMessage {
        padding: 0.5rem 0;
        margin: 0.8rem 0;
    }

    /* Usuario */
    .stChatMessage[data-testid="user-message"] {
        display: flex;
        justify-content: flex-end;
    }

    .stChatMessage[data-testid="user-message"] > div > div {
        background: #C8102E !important;
        color: #ffffff !important;
        border-radius: 16px 16px 2px 16px;
        padding: 1rem 1.2rem !important;
        max-width: 70%;
        box-shadow: 0 2px 8px rgba(200, 16, 46, 0.2);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Asistente */
    .stChatMessage[data-testid="assistant-message"] {
        display: flex;
        justify-content: flex-start;
    }

    .stChatMessage[data-testid="assistant-message"] > div > div {
        background: #ffffff !important;
        color: #000000 !important;
        border-radius: 16px 16px 16px 2px;
        padding: 1rem 1.2rem !important;
        max-width: 75%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* Títulos en mensajes */
    .stChatMessage h3 {
        color: #C8102E !important;
        font-size: 1.05rem !important;
        margin: 0 0 0.5rem 0 !important;
        font-weight: 600 !important;
    }

    .stChatMessage strong {
        color: #000000 !important;
        font-weight: 600;
    }

    .stChatMessage p {
        color: #000000 !important;
        margin: 0.5rem 0 !important;
    }

    .stChatMessage ul, .stChatMessage ol {
        color: #000000 !important;
        margin: 0.5rem 0 !important;
        padding-left: 1.5rem !important;
    }

    .stChatMessage li {
        color: #000000 !important;
        margin: 0.4rem 0 !important;
    }

    /* Avatar */
    .stChatMessage img {
        width: 32px !important;
        height: 32px !important;
        border-radius: 50%;
    }

    /* ========== INPUT ========== */
    .stChatInputContainer {
        background: red;
        border: none;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 -2px 12px rgba(0,0,0,0.08);
        position: sticky;
        bottom: 0;
    }

    .stChatInput > div {
        border-radius: 24px !important;
        border: 2px solid #dee2e6 !important;
        background: #f8f9fa !important;
    }

    .stChatInput > div:focus-within {
        border-color: #C8102E !important;
        background: white !important;
    }

    .stChatInput textarea {
        background: transparent !important;
        border: none !important;
        padding: 0.8rem 1rem !important;
        font-size: 0.95rem !important;
        color: #FFFFFF !important;
    }

    .stChatInput textarea::placeholder {
        color: #6c757d !important;
    }

    .stChatInput button {
        background: #C8102E !important;
        border: none !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        min-width: 40px !important;
        color: white !important;
        transition: all 0.2s;
    }

    .stChatInput button:hover {
        background: #a00d25 !important;
        transform: scale(1.05);
    }

    /* ========== BOTONES ========== */
    .stButton button {
        background: white;
        border: 2px solid #dee2e6;
        color: #495057;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-size: 0.9rem;
        font-weight: 500;
        transition: all 0.2s;
    }

    .stButton button:hover {
        background: #C8102E;
        border-color: #C8102E;
        color: white;
    }

    .stDownloadButton button {
        background: #C8102E !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.3rem !important;
        font-size: 0.9rem !important;
        font-weight: 500;
        margin-top: 0.8rem;
        transition: all 0.2s;
    }

    .stDownloadButton button:hover {
        background: #a00d25 !important;
        transform: translateY(-1px);
    }

    /* ========== FOOTER ========== */
    .footer-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #dee2e6;
    }

    .footer-text {
        font-size: 0.85rem;
        color: #6c757d;
    }

    /* ========== SPINNER ========== */
    .stSpinner > div {
        border-top-color: #C8102E !important;
    }

    /* ========== SCROLLBAR ========== */
    ::-webkit-scrollbar {
        width: 6px;
    }

    ::-webkit-scrollbar-thumb {
        background: #ced4da;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #adb5bd;
    }

    /* ========== RESPONSIVE ========== */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.5rem 1rem 2rem 1rem;
        }

        .header-minimal {
            padding: 0.8rem 1rem;
        }

        .header-text h1 {
            font-size: 1.1rem;
        }

        .header-text p {
            font-size: 0.8rem;
        }

        .stChatMessage[data-testid="user-message"] > div > div,
        .stChatMessage[data-testid="assistant-message"] > div > div {
            max-width: 85%;
        }

        .footer-bar {
            flex-direction: column;
            gap: 0.8rem;
        }
    }

    /* Animación suave */
    .stChatMessage {
        animation: slideIn 0.3s ease;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""", unsafe_allow_html=True)


# ==================== FUNCIONES ====================

@st.cache_resource
def cargar_configuracion():
    try:
        config = {
            'DEEPSEEK_API_KEY': st.secrets["DEEPSEEK_API_KEY"],
            'DEEPSEEK_BASE_URL': "https://api.deepseek.com",
            'MODELO_EMBEDDINGS': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        }
        return config
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

@st.cache_resource(show_spinner=False)
def cargar_documentos(_embeddings):
    docs_dir = Path("data/manuales")
    
    if not docs_dir.exists():
        st.error("No se encuentra la carpeta de manuales")
        st.stop()
    
    all_chunks = []
    
    for pdf_file in docs_dir.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            documentos = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = text_splitter.split_documents(documentos)
            all_chunks.extend(chunks)
        except:
            pass
    
    if not all_chunks:
        st.error("No se pudieron procesar documentos")
        st.stop()
    
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=_embeddings,
        collection_name="manuales_academicos"
    )
    
    return vectorstore

@st.cache_resource(show_spinner=False)
def inicializar_sistema():
    with st.spinner("Inicializando..."):
        config = cargar_configuracion()
        
        os.environ["OPENAI_API_KEY"] = config['DEEPSEEK_API_KEY']
        os.environ["OPENAI_API_BASE"] = config['DEEPSEEK_BASE_URL']
        
        embeddings = HuggingFaceEmbeddings(
            model_name=config['MODELO_EMBEDDINGS'],
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vectorstore = cargar_documentos(embeddings)
        
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3,
            openai_api_base=config['DEEPSEEK_BASE_URL'],
            openai_api_key=config['DEEPSEEK_API_KEY']
        )
        
        template_prompt = """Eres un asistente virtual de la Secretaría Académica de la Universidad Libre.
Tu objetivo es responder preguntas sobre procesos académicos basándote ÚNICAMENTE en el contexto proporcionado.

Reglas importantes:
1. Si la información NO está en el contexto, di "No encuentro esa información en los manuales disponibles"
2. Sé claro, conciso y profesional
3. Cita el manual o sección cuando sea relevante
4. Si hay pasos, enuméralos claramente
5. Usa un lenguaje amigable y cercano
6. ten en cuenta que es muy probable que haya manuales que no tengan listas enumeradas sino que lo digan todo en un párrafo

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA:"""
        
        prompt = PromptTemplate(
            template=template_prompt,
            input_variables=["context", "question"]
        )
        
        return vectorstore, llm, prompt

class GestorFormatos:
    def __init__(self):
        self.formatos = self._cargar_catalogo()
        self.formatos_dir = Path("data/formatos_descargables")
    
    def _cargar_catalogo(self):
        catalogo_path = Path("data/formatos_descargables/catalogo_formatos.json")
        try:
            with open(catalogo_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data['formatos']
        except:
            return []
    
    def detectar_formato(self, pregunta):
        pregunta_lower = pregunta.lower()
        palabras = ['formato', 'formulario', 'solicitud', 'plantilla', 'documento']
        
        if not any(p in pregunta_lower for p in palabras):
            return None
        
        for formato in self.formatos:
            for keyword in formato['keywords']:
                if keyword.lower() in pregunta_lower:
                    return formato
        return None
    
    def obtener_archivo(self, formato):
        ruta = self.formatos_dir / formato['archivo']
        return ruta if ruta.exists() else None

def consultar_rag(pregunta, vectorstore, llm, prompt, gestor_formatos):
    formato_detectado = gestor_formatos.detectar_formato(pregunta)
    
    if formato_detectado:
        archivo = gestor_formatos.obtener_archivo(formato_detectado)
        respuesta = f"### {formato_detectado['nombre']}\n\n{formato_detectado['descripcion']}\n\n"
        
        if archivo:
            respuesta += "**Instrucciones:**\n" + formato_detectado['instrucciones']
        else:
            respuesta += "El archivo no está disponible."
        
        return {"respuesta": respuesta, "archivo": archivo}
    
    docs = vectorstore.similarity_search(pregunta, k=3)
    contexto = "\n\n".join([doc.page_content for doc in docs])
    prompt_completo = prompt.format(context=contexto, question=pregunta)
    respuesta_obj = llm.invoke([HumanMessage(content=prompt_completo)])
    
    return {"respuesta": respuesta_obj.content, "archivo": None}

# ==================== INTERFAZ ====================

def main():
    # Header minimalista
    logo_path = Path("logo.png")
    
    st.markdown('<div class="header-minimal">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 10])
    with col1:
        if logo_path.exists():
            st.image("logo.png", width=50)
    with col2:
        st.markdown("""
            <div class="header-text">
                <h1>Secretaría Académica</h1>
                <p>Universidad Libre</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Inicializar
    try:
        vectorstore, llm, prompt = inicializar_sistema()
        gestor_formatos = GestorFormatos()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
    
    # Historial
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "¡Hola! Soy Librecito, tu asistente virtual. ¿En qué puedo ayudarte?"
        })
    
    # Chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message.get("archivo"):
                try:
                    with open(message["archivo"], "rb") as f:
                        st.download_button(
                            label="📥 Descargar formato",
                            data=f.read(),
                            file_name=message['archivo'].name,
                            mime="application/octet-stream",
                            key=f"dl_{message.get('timestamp', '')}"
                        )
                except:
                    pass
    
    # Input
    if prompt_input := st.chat_input("Escribe tu pregunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        
        with st.chat_message("user"):
            st.markdown(prompt_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    time.sleep(0.3)
                    resultado = consultar_rag(prompt_input, vectorstore, llm, prompt, gestor_formatos)
                    
                    st.markdown(resultado["respuesta"])
                    
                    if resultado["archivo"]:
                        try:
                            with open(resultado["archivo"], "rb") as f:
                                st.download_button(
                                    label="📥 Descargar formato",
                                    data=f.read(),
                                    file_name=resultado['archivo'].name,
                                    mime="application/octet-stream",
                                    key=f"dl_new_{time.time()}"
                                )
                        except:
                            pass
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": resultado["respuesta"],
                        "archivo": resultado["archivo"],
                        "timestamp": time.time()
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Footer
    st.markdown('<div class="footer-bar">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<span class="footer-text">Universidad Libre © 2025</span>', unsafe_allow_html=True)
    with col2:
        if st.button("Limpiar"):
            st.session_state.messages = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()