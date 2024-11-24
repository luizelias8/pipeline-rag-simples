# https://www.youtube.com/watch?v=tcqEUSNCn8I&t=1s
# https://github.com/pixegami/langchain-rag-tutorial/blob/main/query_data.py

import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings # Importa a funcionalidade de embeddings da OpenAI
from langchain_community.vectorstores import Chroma

# Caminho onde o banco de dados Chroma será armazenado
CAMINHO_CHROMA = 'chroma'

# Caminho onde os dados dos livros (arquivos .txt) estão armazenados
CAMINHO_DADOS = 'dados/transcricoes'

# Carregar variáveis de ambiente. Assume que o projeto contém um arquivo .env com as chaves da API
load_dotenv()

def carregar_documentos():
    # Usando o carregador de diretórios para ler arquivos com a extensão .txt
    carregador = DirectoryLoader(CAMINHO_DADOS, glob='*.txt')
    # Carrega os documentos do diretório especificado
    documentos = carregador.load()
    return documentos

def dividir_texto(documentos):
    # Instancia um divisor de texto que divide o conteúdo dos documentos em partes
    divisor_texto = RecursiveCharacterTextSplitter(
        chunk_size=300, # Tamanho máximo de cada parte de texto
        chunk_overlap=100, # Quantidade de sobreposição entre as partes
        # Isso ajuda a garantir que o contexto de uma parte do texto não seja perdido na transição para a próxima parte.
        length_function=len, # Função para medir o tamanho do texto
        add_start_index=True # Adiciona um índice de início às partes
    )
    # O RecursiveCharacterTextSplitter é uma classe fornecida pelo pacote LangChain que serve para dividir documentos de texto em partes menores (ou "chunks") de forma recursiva.
    # O RecursiveCharacterTextSplitter divide o texto de acordo com algumas regras definidas, como o tamanho máximo de cada parte e a sobreposição entre as partes.
    # O processo de divisão é recursivo, o que significa que, se o tamanho de um "chunk" atingir o limite definido, ele será dividido em partes menores, se necessário.

    # Divide os documentos em partes menores
    partes = divisor_texto.split_documents(documentos)
    print(f"Divididos {len(documentos)} documentos em {len(partes)} partes.")
    # Dividir o texto em partes (ou chunks) tem várias vantagens e é um procedimento comum em processos de processamento de linguagem natural
    # e análise de grandes volumes de dados.
    # Muitos modelos de linguagem têm uma limitação de tamanho de entrada (tokens).
    # Dividir o texto em partes menores permite que você lide com textos grandes sem ultrapassar esse limite.
    # Processar grandes volumes de texto de uma vez pode sobrecarregar a memória do sistema e afetar a performance.
    # Dividir o texto ajuda a reduzir esse impacto.
    # Dividir o texto em partes facilita a paralelização do processo.
    # Ou seja, diferentes partes do texto podem ser processadas simultaneamente em máquinas diferentes ou threads separadas, aumentando a velocidade de processamento.
    # Ao dividir os documentos em partes, o modelo pode criar embeddings (representações numéricas) mais precisos para cada parte do texto.
    # Em vez de tentar gerar um embedding para um documento grande, você gera embeddings específicos para cada parte, o que pode levar a resultados mais precisos quando essas partes são consultadas.
    # Modelos de linguagem tendem a ser mais eficazes ao processar textos com contextos menores.
    # Isso significa que, ao dividir um texto em partes menores, cada parte pode ser analisada em seu contexto, sem perder informações cruciais.

    # Exibe o conteúdo da décima parte do documento
    documento = partes[10]
    print(documento.page_content)
    print(documento.metadata)
    # O campo metadata é usado para armazenar informações adicionais relacionadas
    # a um documento ou a uma parte de um documento, que não fazem parte do conteúdo textual principal.
    # Ele serve como um metadado ou informação adicional que pode ser útil para rastrear, organizar ou fornecer contexto extra sobre o conteúdo do documento ou "chunk".

    return partes

def salvar_no_chroma(partes):
    # Limpar o banco de dados existente (se houver) antes de salvar os novos dados
    if os.path.exists(CAMINHO_CHROMA):
        shutil.rmtree(CAMINHO_CHROMA)

    # Criar um novo banco de dados a partir das partes dos documentos usando embeddings da OpenAI
    db = Chroma.from_documents(partes, OpenAIEmbeddings(), persist_directory=CAMINHO_CHROMA)
    # Persistir os dados no Chroma (salvamento)
    # db.persist()
    # A partir da versão 0.4.x do Chroma, o método manual de persistência, como o uso do método db.persist(), não é mais necessário, pois o Chroma agora persiste automaticamente os dados no diretório especificado ao criar o banco de dados.
    print(f"Salvou {len(partes)} partes em {CAMINHO_CHROMA}.")

def gerar_armazenamento_de_dados():
    # Carregar os documentos, dividir o texto em partes e salvar no Chroma
    documentos = carregar_documentos()
    partes = dividir_texto(documentos)
    salvar_no_chroma(partes)

def principal():
    # Função principal que chama o processo de geração do armazenamento de dados
    gerar_armazenamento_de_dados()

# Se este script for executado diretamente, chama a função principal
if __name__ == '__main__':
    principal()