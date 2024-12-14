import spacy
import psycopg2
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Carica il modello spaCy
nlp = spacy.load("en_core_web_md")

# Confronta semanticamente le risposte e conteggia le risposte migliori
def calculate_semantic_similarity(ground_truth, answer):
    gt_embedding = nlp(ground_truth).vector
    ans_embedding = nlp(answer).vector
    return cosine_similarity([gt_embedding], [ans_embedding])[0][0]

# Impostare la connessione al database
def connect_to_db():
    return psycopg2.connect(
        host="localhost",  
        database="",  
        user="",  
        password=""  
    )

def count_correct_responses(threshold=0.5):
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Recupera le risposte dalla tabella qa_rag_ontology
    cursor.execute("SELECT ground_truth, answer_llama, answer_falcon, answer_orca FROM pdf_esempionedb")
    results = cursor.fetchall()
    
    # Contatori per le risposte semanticamente corrette
    correct_counts = Counter()

    for row in results:
        ground_truth, answer_llama, answer_falcon, answer_orca = row
        
        # Calcola similarità semantica per ciascuna risposta
        sim_llama = calculate_semantic_similarity(ground_truth, answer_llama)
        sim_falcon = calculate_semantic_similarity(ground_truth, answer_falcon)
        sim_orca = calculate_semantic_similarity(ground_truth, answer_orca)

        # Trova la similarità più alta e verifica con la soglia
        max_sim = max(sim_llama, sim_falcon, sim_orca)
        if max_sim >= threshold:
            if max_sim == sim_llama:
                correct_counts['llama'] += 1
            elif max_sim == sim_falcon:
                correct_counts['falcon'] += 1
            elif max_sim == sim_orca:
                correct_counts['orca'] += 1

    cursor.close()
    conn.close()

    # Stampa il contatore dei risultati corretti
    print("Counter best answers (above the threshold):")
    for model, count in correct_counts.items():
        print(f"{model.capitalize()}: {count}")

count_correct_responses(threshold=0.5)
