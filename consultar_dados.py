# https://www.youtube.com/watch?v=tcqEUSNCn8I&t=1s
# https://github.com/pixegami/langchain-rag-tutorial/blob/main/query_data.py

import os
# Biblioteca para lidar com argumentos de linha de comando
import argparse
# Importa a classe Chroma para manipulação do banco de dados vetorial
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
# Importa a funcionalidade de embeddings da OpenAI
from langchain_openai import OpenAIEmbeddings
# Importa o modelo de chat da OpenAI
from langchain_openai import ChatOpenAI
# Importa o modelo de chat da Groq
from langchain_groq import ChatGroq
# Importa a funcionalidade para criação de prompts personalizados
from langchain.prompts import ChatPromptTemplate

# Caminho onde o banco de dados vetorial Chroma será armazenado
CAMINHO_CHROMA = 'chroma'

# Template do prompt utilizado para gerar respostas com base no contexto fornecido
TEMPLATE_PROMPT = """
Responda à pergunta com base apenas no seguinte contexto:

{contexto}

---

Resposta à pergunta com base no contexto acima: {pergunta}
"""

# Função principal que coordena o fluxo do programa
def principal():
    # Configura a interface de linha de comando (CLI).
    parser = argparse.ArgumentParser()
    # Adiciona um argumento para receber o texto da consulta (query) do usuário
    parser.add_argument('texto_consulta', type=str, help='O texto da consulta.')

    # Analisa os argumentos fornecidos pelo usuário
    args = parser.parse_args()
    texto_consulta = args.texto_consulta # Armazena o texto da consulta

    # Configura o banco de dados vetorial (Chroma) com embeddings da OpenAI
    funcao_embedding = OpenAIEmbeddings()
    db = Chroma(persist_directory=CAMINHO_CHROMA, embedding_function=funcao_embedding)

    # Realiza a busca no banco de dados com base na similaridade.
    resultados = db.similarity_search_with_relevance_scores(texto_consulta, k=3)

    # Verifica se encontrou resultados relevantes com pontuação mínima de 0.7
    if len(resultados) == 0 or resultados[0][1] < 0.7:
        print("Não foi possível encontrar resultados correspondentes.")
        return

    # for resultado, score in resultados:
    #     print(f'{resultado.page_content}\n')

    # Concatena os conteúdos das páginas retornadas como contexto
    texto_contexto = '\n\n---\n\n'.join([resultado.page_content for resultado, score in resultados])
    # print(contexto)

    # Cria o prompt personalizado usando o contexto e a pergunta
    template_prompt = ChatPromptTemplate.from_template(TEMPLATE_PROMPT)
    # O método ChatPromptTemplate.from_template() aceita um texto de template como parâmetro.
    # Esse texto deve conter placeholders que podem ser substituídos dinamicamente ao formatar o prompt.

    # # Substitui os placeholders pelo contexto e pergunta fornecidos
    prompt = template_prompt.format(contexto=texto_contexto, pergunta=texto_consulta)
    # O método format é utilizado posteriormente para substituir os placeholders pelos valores reais.
    # print(prompt)

    # # Se não quiser usar um template de prompt, pode construir o texto do prompt diretamente como uma string formatada
    # prompt = f"""
    # Responda à pergunta com base apenas no seguinte contexto:

    # {texto_contexto}

    # ---

    # Resposta à pergunta com base no contexto acima: {texto_consulta}
    # """

    # Configura o modelo de linguagem com a API da Groq
    modelo = ChatGroq(
        model='llama-3.1-70b-versatile', # Nome do modelo utilizado
        # temperature=0, # Controla a criatividade do modelo
        api_key=os.getenv('GROQ_API_KEY') # Recupera a chave de API do modelo Groq das variáveis de ambiente
    )

    # Envia o prompt ao modelo e obtém a resposta
    resposta = modelo.invoke(prompt)
    print(resposta.content) # Exibe a resposta ao usuário

# Executa a função principal se o script for chamado diretamente
if __name__ == '__main__':
    principal()