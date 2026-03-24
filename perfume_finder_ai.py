import streamlit as st
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss

# ----------------------------------
# CARREGAR BASE DE PERFUMES
# ----------------------------------

# dataset pode ter até 50k perfumes
df = pd.read_csv("perfumes_dataset.csv")

# colunas esperadas:
# nome
# marca
# descricao
# notas
# genero
# estacao
# preco

# ----------------------------------
# MODELO DE EMBEDDINGS
# ----------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# gerar embeddings
embeddings = model.encode(
    df["descricao"].tolist(),
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

# ----------------------------------
# INDEXAÇÃO COM FAISS
# ----------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# ----------------------------------
# FUNÇÕES
# ----------------------------------

def buscar_similares(nome, k=10):

    idx = df[df["nome"] == nome].index[0]

    vetor = embeddings[idx].reshape(1, -1)

    distancias, indices = index.search(vetor, k+1)

    resultado = df.iloc[indices[0][1:]]

    return resultado[["nome","marca","preco"]]


def encontrar_clones_baratos(nome):

    idx = df[df["nome"] == nome].index[0]

    preco_original = df.iloc[idx]["preco"]

    vetor = embeddings[idx].reshape(1,-1)

    distancias, indices = index.search(vetor, 30)

    candidatos = df.iloc[indices[0]]

    clones = candidatos[
        candidatos["preco"] < preco_original * 0.6
    ]

    return clones[["nome","marca","preco"]].head(5)


def recomendar_por_nota(nota):

    resultado = df[
        df["notas"].str.contains(
            nota,
            case=False,
            na=False
        )
    ]

    return resultado[["nome","marca","notas"]].head(20)


def recomendar_por_estacao(estacao):

    return df[
        df["estacao"] == estacao
    ][["nome","marca"]].head(20)


def recomendar_por_genero(genero):

    return df[
        df["genero"] == genero
    ][["nome","marca"]].head(20)


def comparar_perfumes(p1, p2):

    idx1 = df[df["nome"] == p1].index[0]
    idx2 = df[df["nome"] == p2].index[0]

    v1 = embeddings[idx1].reshape(1,-1)
    v2 = embeddings[idx2].reshape(1,-1)

    sim = np.dot(v1,v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2))

    return float(sim)


# ----------------------------------
# IA TIPO NETFLIX (RECOMENDAÇÃO)
# ----------------------------------

def recomendar_usuario(perfumes_usuario):

    vetores = []

    for p in perfumes_usuario:

        idx = df[df["nome"] == p].index[0]

        vetores.append(embeddings[idx])

    media = np.mean(vetores, axis=0).reshape(1,-1)

    distancias, indices = index.search(media, 10)

    return df.iloc[indices[0]][["nome","marca"]]


# ----------------------------------
# INTERFACE DO SITE
# ----------------------------------

st.title("Perfume Finder AI")

st.write("Descubra perfumes similares, clones baratos e recomendações inteligentes")

perfume = st.selectbox(
    "Escolha um perfume",
    df["nome"]
)

if st.button("Perfumes similares"):

    st.write(buscar_similares(perfume))


if st.button("Clones baratos"):

    st.write(encontrar_clones_baratos(perfume))


st.subheader("Buscar por nota")

nota = st.text_input("Ex: baunilha, oud, citrico")

if st.button("Buscar nota"):

    st.write(recomendar_por_nota(nota))


st.subheader("Recomendar por estação")

estacao = st.selectbox(
    "Estação",
    ["verao","primavera","outono","inverno"]
)

if st.button("Recomendar"):

    st.write(recomendar_por_estacao(estacao))


st.subheader("Recomendar por gênero")

genero = st.selectbox(
    "Genero",
    ["masculino","feminino","unissex"]
)

if st.button("Buscar genero"):

    st.write(recomendar_por_genero(genero))


st.subheader("Comparar perfumes")

p1 = st.selectbox("Perfume 1", df["nome"])
p2 = st.selectbox("Perfume 2", df["nome"], index=1)

if st.button("Comparar"):

    st.write("Similaridade:", round(comparar_perfumes(p1,p2),2))


st.subheader("Recomendação estilo Netflix")

favoritos = st.multiselect(
    "Perfumes que você gosta",
    df["nome"]
)

if st.button("Recomendar para mim"):

    if len(favoritos) > 0:

        st.write(recomendar_usuario(favoritos))
