import time
import json
import psycopg2
from gpt4all import GPT4All
from wikipediaapi import Wikipedia

# Configurazione dei percorsi dei modelli
# 1. Carica i modelli locali
model_orca = GPT4All(r"C:\Users\nicol\gpt4all\resources\orca-2-7b.Q4_0.gguf")
model_llama = GPT4All(r"C:\Users\nicol\gpt4all\resources\Meta-Llama-3-8B-Instruct.Q4_0.gguf")
model_falcon = GPT4All(r"C:\Users\nicol\gpt4all\resources\gpt4all-falcon-newbpe-q4_0.gguf")

# Classe wrapper per i modelli LLM
class LocalLLM:
    def __init__(self, model):
        self.model = model

    def __call__(self, prompt):
        return self.model.generate(prompt)

# Inizializzazione dei modelli
falcon = LocalLLM(model_falcon)
llama = LocalLLM(model_llama)
orca = LocalLLM(model_orca)

# Configurazione del retriever di Wikipedia
class WikipediaRetriever:
    def __init__(self):
        self.wiki = Wikipedia(language='en', user_agent="RAG_project/1.0 (nicoladesiena@outlook.it)")

    def retrieve(self, query):
        page = self.wiki.page(query)
        if page.exists():
            return page.summary
        else:
            return "No relevant information found."

retriever = WikipediaRetriever()

# Classe per la catena di RAG
class RetrievalQA:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def __call__(self, query):
        context = self.retriever.retrieve(query)
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        return self.llm(prompt)

# Inizializzazione delle catene di RAG per ogni modello
qa_falcon = RetrievalQA(falcon, retriever)
qa_llama = RetrievalQA(llama, retriever)
qa_orca = RetrievalQA(orca, retriever)

# Configurazione della connessione al database PostgreSQL
def connect_to_db():
    return psycopg2.connect(
        host="localhost",
        database="Esempionefinale",
        user="postgres",
        password="nicola"
    )

# Funzione principale
def main():
    # Caricamento delle domande dal file JSON
    with open(r"C:\Users\nicol\Desktop\progetto_tesi\Esempione_finale\esempione_def.json") as file:
        questions = json.load(file)

    # Inizializzazione della connessione al database
    conn = connect_to_db()
    cur = conn.cursor()

    for idx, question in enumerate(questions):
        id = question["id"]
        question_text = question["question"]
        ground_truth = question.get("ground_truth", "")

        print(f"Processing question {idx + 1}/{len(questions)}: {question_text}")

        # Generazione delle risposte e misurazione dei tempi
        start_time = time.time()
        answer_falcon = qa_falcon(question_text)
        time_response_falcon = time.time() - start_time

        start_time = time.time()
        answer_llama = qa_llama(question_text)
        time_response_llama = time.time() - start_time

        start_time = time.time()
        answer_orca = qa_orca(question_text)
        time_response_orca = time.time() - start_time

        # Salvataggio delle risposte nel database
        save_to_db(cur, id, question_text, ground_truth, answer_orca, answer_llama, answer_falcon, time_response_orca, time_response_llama, time_response_falcon)
        conn.commit()

        # Mostra il progresso
        print(f"Saved responses for question {idx + 1}/{len(questions)}")

    # Chiusura delle risorse
    cur.close()
    conn.close()

# Funzione to save results in database
def save_to_db(cur, id, question, ground_truth, answer_orca, answer_llama, answer_falcon, time_orca, time_llama, time_falcon):
    cur.execute("""
        INSERT INTO wiki_esempionedb (id, question, ground_truth, answer_orca, answer_llama, answer_falcon, time_orca, time_llama, time_falcon)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
    """, (id, question, ground_truth, answer_orca, answer_llama, answer_falcon, time_orca, time_llama, time_falcon))

if __name__ == "__main__":
    main()
