import pandas as pd
import json
import time
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import psycopg2
from gpt4all import GPT4All

# Impostare la connessione al database
def connect_to_db():
    return psycopg2.connect(
        host="localhost",  # Inserisci nome host
        database="",  # Inserisci il nome del database
        user="",  # Inserisci il nome utente
        password=""  # Inserisci la password
    )

# Carica i due file CSV locali
planets_df = pd.read_csv(r"C:\Users\nicol\Desktop\progetto_tesi\Esempione_finale\Database\planets.csv")
satellites_df = pd.read_csv(r"C:\Users\nicol\Desktop\progetto_tesi\Esempione_finale\Database\satellites.csv")

# 1. Carica i modelli locali
model_orca = GPT4All(r"C:\Users\nicol\gpt4all\resources\orca-2-7b.Q4_0.gguf")
model_llama = GPT4All(r"C:\Users\nicol\gpt4all\resources\Meta-Llama-3-8B-Instruct.Q4_0.gguf")
model_falcon = GPT4All(r"C:\Users\nicol\gpt4all\resources\gpt4all-falcon-newbpe-q4_0.gguf")

# Configura il prompt
prompt_template = PromptTemplate(template="Domanda: {question}\nContesto: {context}\nRisposta:", input_variables=["question", "context"])

# Funzione per il retrieval dei dati
def retrieve_context(question):
    keywords = question.lower().split()  # Suddivide la domanda in parole chiave
    # Filtro per ogni file CSV in base ai campi che contengono keywords
    planets_results = planets_df[planets_df.apply(lambda row: any(keyword in row.astype(str).str.lower().values for keyword in keywords), axis=1)]
    satellites_results = satellites_df[satellites_df.apply(lambda row: any(keyword in row.astype(str).str.lower().values for keyword in keywords), axis=1)]
    context = pd.concat([planets_results, satellites_results]).to_string(index=False)
    return context if context else "No relevant information found."

# Funzione principale per generare risposte
def generate_responses(question):
    context = retrieve_context(question)
    prompt = prompt_template.format(question=question, context=context)

    responses = {}
    start_time = time.time()
    responses["orca"] = model_orca.generate(prompt)
    responses["time_orca"] = time.time() - start_time

    start_time = time.time()
    responses["llama"] = model_llama.generate(prompt)
    responses["time_llama"] = time.time() - start_time

    start_time = time.time()
    responses["falcon"] = model_falcon.generate(prompt)
    responses["time_falcon"] = time.time() - start_time

    return responses

# Carica il file JSON con le domande
with open(r"C:\Users\nicol\Desktop\progetto_tesi\Esempione_finale\esempione_def.json") as f:
    questions = json.load(f)

# Itera sulle domande e salva i risultati
for question_item in questions:
    question_id = question_item["id"]  # Estrae l'ID della domanda
    question = question_item["question"].strip()  # Estrae il testo e rimuove eventuali spazi bianchi
    ground_truth = question_item["ground_truth"]  # Estrae la ground truth
    responses = generate_responses(question)

    # Salva le risposte nel database
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO ragdb_esempione (question_id, question, ground_truth, answer_llama, response_time_llama, 
                                        answer_orca, response_time_orca, 
                                        answer_falcon, response_time_falcon) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (question_id, question, ground_truth, 
                  responses.get("llama"), responses.get("time_llama"),
                  responses.get("orca"), responses.get("time_orca"),
                  responses.get("falcon"), responses.get("time_falcon")))
            conn.commit()

    print(f"Domanda ID: {question_id} - Risposte salvate.")

print("Processo completato e risultati salvati nel database PostgreSQL.")
