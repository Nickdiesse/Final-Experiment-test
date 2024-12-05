import pandas as pd
import json
import time
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import psycopg2
from gpt4all import GPT4All
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer

# Funzione per salvare gradualmente le risposte nel database
def save_to_db(result):
    conn = psycopg2.connect(
        host="localhost",
        database="Esempionefinale",
        user="postgres",
        password="nicola"
    )
    cursor = conn.cursor()

    # Inserimento dati nel database
    cursor.execute("""
        INSERT INTO pdf_esempionedb (question_id, question_text, ground_truth, answer_llama, answer_orca, answer_falcon, llama_time, orca_time, falcon_time)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
    """, (
        result["question_id"],
        result["question_text"],
        result["ground_truth"],
        result["answer_llama"],
        result["answer_orca"],
        result["answer_falcon"],
        result["llama_time"],
        result["orca_time"],
        result["falcon_time"]
    ))

    # Esegui il commit e chiudi la connessione
    conn.commit()
    cursor.close()
    conn.close()

# Caricamento del singolo file PDF e creazione di un indice di vettori
def create_pdf_index(pdf_file):
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = FAISS.from_documents(documents, embeddings)
    return index

pdf_file_path = r"C:\Users\nicol\Desktop\progetto_tesi\Esempione_finale\PDF\Astronomy-For-Mere-Mortals-v-23.pdf"
pdf_index = create_pdf_index(pdf_file_path)

# Caricamento delle domande dal file JSON
with open(r"C:\Users\nicol\Desktop\progetto_tesi\Esempione_finale\esempione_def.json") as f:
    questions = json.load(f)

# Carica i modelli .gguf
models = {
    "Llama": GPT4All(r"C:\Users\nicol\gpt4all\resources\Meta-Llama-3-8B-Instruct.Q4_0.gguf"),
    "Orca": GPT4All(r"C:\Users\nicol\gpt4all\resources\orca-2-7b.Q4_0.gguf"),
    "Falcon": GPT4All(r"C:\Users\nicol\gpt4all\resources\gpt4all-falcon-newbpe-q4_0.gguf")
}

# Carica il modello e il tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

def ask_question_to_models(models, question, pdf_index):
    context = pdf_index.similarity_search(question, k=3)
    context_text = " ".join([doc.page_content for doc in context])

    # Tokenizza il contesto per verificare la lunghezza
    context_tokens = tokenizer(context_text)["input_ids"]

    # Limita il numero di token totali (per esempio, 2048 - lunghezza della domanda)
    max_context_tokens = 1800  # Lascia un margine per la domanda
    if len(context_tokens) > max_context_tokens:
        # Ritaglia il contesto per non eccedere i 1800 token
        context_text = tokenizer.decode(context_tokens[:max_context_tokens])

    # Fai domanda a ogni modello
    responses = {}
    response_times = {}  # Dizionario per salvare i tempi di risposta
    for model_name, model in models.items():
        prompt = f"Context: {context_text}\nQuestion: {question}\nAnswer:"

        # Misurazione del tempo di inizio
        start_time = time.time()

        # Utilizza il metodo generate per ottenere la risposta
        response = model.generate(prompt)

        # Misurazione del tempo di fine
        end_time = time.time()

        # Calcola il tempo di risposta
        response_time = end_time - start_time
        response_times[model_name] = response_time

        # Verifica se la risposta è una stringa o un dizionario
        if isinstance(response, str):
            responses[model_name] = response
        elif isinstance(response, dict) and "choices" in response:
            responses[model_name] = response["choices"][0]["text"]
        else:
            responses[model_name] = "Error: Unexpected response format"

        # Stampa il progresso della risposta e il tempo impiegato
        print(f"Model: {model_name} - Question: {question[:30]}... - Response: {responses[model_name][:50]}... - Time: {response_time:.2f} seconds")

    return responses, response_times  # Ritorna anche i tempi di risposta

start_id = 43  # ID di partenza

for question in questions:
    q_id = question["id"]
    if q_id < start_id:
        continue  # Salta le domande con ID minore di start_id
    
    q_text = question["question"]
    ground_truth = question["ground_truth"]
    print(f"Processing Question ID: {q_id} - {q_text[:50]}...")  # Messaggio di inizio per ogni domanda

    # Ottieni le risposte dai modelli e i tempi di risposta
    responses, response_times = ask_question_to_models(models, q_text, pdf_index)

    # Prepara il dizionario per salvare i risultati di ogni modello
    result = {
        "question_id": q_id,
        "question_text": q_text,
        "ground_truth": ground_truth,
        "answer_llama": responses.get("Llama", ""),
        "answer_orca": responses.get("Orca", ""),
        "answer_falcon": responses.get("Falcon", ""),
        "llama_time": response_times.get("Llama", 0),
        "orca_time": response_times.get("Orca", 0),
        "falcon_time": response_times.get("Falcon", 0)
    }

    # Salva la risposta e i tempi nel database per la domanda corrente
    save_to_db(result)
    print(f"Saved response and times for question ID {q_id} to the database.")

print("Processo completato e risultati salvati nel database PostgreSQL.")
