import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import torch
from langchain_huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub


from dotenv import load_dotenv

load_dotenv()

# Configurações do Streamlit
st.set_page_config(page_title="Seu assistente virtual 🤖", page_icon="🤖")
st.title("Seu assistente virtual 🤖")

model_class = "hf_hub" # @param ["hf_hub", "openai", "ollama"]

# Função de Carregamento dos Modelos

def model_hf_hub(model = "meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
  llm = HuggingFaceHub(repo_id=model,
                       model_kwargs={
                          "temperature":temperature,
                          "return_full_text": False,
                          "max_new_tokens": 512,  # n° max de tokens gerados
                          })
  return llm

def model_openai(model = "gpt-4o-mini", temperature=0.1):
  llm = ChatOpenAI(model= model, temperature=temperature)
  return llm

def model_ollama(model = "phi3", temperature=0.1):
  llm = ChatOllama(model=model, temperature=temperature)
  return llm

# Função para fazer conversa com chatbot

def model_response(user_query, chat_history, model_class):  # Usuário solicitar, histórico das conversas,

    # Carregamento da LLM
    if model_class == "hf_hub":
      llm = model_hf_hub()
    elif model_class == "openai":
      llm = model_openai()
    elif model_class == "ollama":
      llm = model_ollama()

    # Definição do Prompt
    system_prompt = """
        Você é um assistente prestativo e está respondendo perguntas gerais. Responda em português."
    """

    # Adequando Pipeline
    if model_class.startswith("hf"):
        user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>" # modelo padrão do hugfface que recebe o input
    else:
        user_prompt = "{input}" # Valor padrão para outros modelos

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt), # Recebe o prompt
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", user_prompt)      # Recebe a pergunta do usuário
    ])

    # Criação da chain
    # StrOutputParser(): faz a formatação final do prompt

    chain = prompt_template | llm | StrOutputParser()

    # Retorno da resposta
    return chain.stream({"chat_history": chat_history,"input": user_query})

    # chain.invoke(): retorna a resposta quando todo processamento for concluído
    # chain.stream(): o texto será mostrado a medida que é gerado

# Controla as mensagens trocadas com o chatbot, pois depois de um tempo sao apagadas do chache se nao usar o chatbot

if "chat_history" not in st.session_state: # Verifica se a variável chat_history tem algum valor. se sim a sessão está ativa e já estamos nos comunicando com chatbot
    st.session_state.chat_history = [      # Se não vamos criar outra cessao e guando abrir o site o chat vai da uma mensagem de boas vindas
        AIMessage(content="Olá, sou o seu assistente virtual! Como posso ajudar você?"),
    ]

# Definição da conversa (renderização do histórico de mensagens)
for message in st.session_state.chat_history: # Acessa o histórico de mensagens, nela fica armazenadas as mensagens do usuário e da IA
    if isinstance(message, AIMessage): # Verifica se a mesagem é um objeto da classe IAMessage, todas elas vão ser respostas da IA
        with st.chat_message("AI"):
            st.write(message.content) # Escreve as mesagens na tela
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Digite sua mensagem aqui...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)   # Formata exibe a msg na interface do st

    with st.chat_message("AI"):
        resp = st.write_stream(model_response(user_query, st.session_state.chat_history, model_class))
        print(st.session_state.chat_history)

    st.session_state.chat_history.append(AIMessage(content=resp))
