import streamlit as st
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

st.set_page_config(page_title="Modelo de Conhecimento Bíblico", layout="centered")

@st.cache_resource
def carregar_tudo():
    # 1. Lê a Bíblia
    with open("biblia.txt", "r", encoding="utf-8") as f:
        texto = f.read()
    pedacos = [texto[i:i+1000] for i in range(0, len(texto), 1000)]
    
    # 2. Cria a memória (Embeddings)
    modelo_emb = SentenceTransformer('all-MiniLM-L6-v2')
    emb_biblia = modelo_emb.encode(pedacos, show_progress_bar=False)
    
    # 3. Carrega o cérebro (Qwen 0.5B)
    model_id = "Qwen/Qwen2.5-0.5B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # AQUI ESTÁ A CORREÇÃO: Removemos o device_map="cpu"
    modelo_llm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    
    pipe = pipeline("text-generation", model=modelo_llm, tokenizer=tokenizer, max_new_tokens=150)
    
    return pedacos, modelo_emb, emb_biblia, pipe

with st.spinner("Carregando a Bíblia e o cérebro da IA... (isso pode demorar um pouco)"):
    pedacos, modelo_emb, emb_biblia, pipe = carregar_tudo()

st.title("Modelo de Conhecimento Bíblico")
st.write("Pergunte qualquer coisa sobre a Bíblia. Respostas rápidas e diretas.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Digite sua pergunta aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            # Busca na Bíblia
            emb_pergunta = modelo_emb.encode([prompt])[0]
            similaridades = np.dot(emb_biblia, emb_pergunta) / (np.linalg.norm(emb_biblia, axis=1) * np.linalg.norm(emb_pergunta))
            top_3 = np.argsort(similaridades)[-3:][::-1]
            contexto = "\n".join([pedacos[i] for i in top_3])
            
            prompt_final = f"Responda à pergunta usando APENAS o contexto bíblico abaixo.\nContexto: {contexto}\nPergunta: {prompt}\nResposta:"
            
            resultado = pipe(prompt_final)
            resposta = resultado[0]['generated_text'].split("Resposta:")[-1].strip()
            
            st.markdown(resposta)
    st.session_state.messages.append({"role": "assistant", "content": resposta})
