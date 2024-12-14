import psycopg2
import re

def calculate_f1(predicted, ground_truth):
    # Tokenizza il testo dividendo su spazi e caratteri speciali
    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    pred_tokens = tokenize(predicted)
    gt_tokens = tokenize(ground_truth)

    # Calcola l'F1-score
    common_tokens = set(pred_tokens) & set(gt_tokens)
    num_common = len(common_tokens)

    if num_common == 0:
        f1 = 0.0
    else:
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)



    return f1

# Connessione al database PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="",
    user="",
    password=""
)
cursor = conn.cursor()


# Estrai i dati dalla tabella esistente
cursor.execute("SELECT ground_truth, answer_llama, answer_orca, answer_falcon FROM esempione_prompt_db;")
rows = cursor.fetchall()

# Calcola F1-score per ogni modello e salva nella nuova tabella
for row in rows:
    ground_truth = row[1]
    answers = {
        "llama": row[1],
        "orca": row[2],
        "falcon": row[3]
    }

    # Calcola F1-score ed Exact Match per ciascun modello
    f1_llama = calculate_f1(answers["llama"], ground_truth)
    f1_orca = calculate_f1(answers["orca"], ground_truth)
    f1_falcon = calculate_f1(answers["falcon"], ground_truth)

    # Inserisci i risultati nella tabella model_evaluation
    cursor.execute("""
        INSERT INTO f1score_prompt_es(f1_llama, f1_orca, f1_falcon)
        VALUES (%s, %s, %s);
    """, (f1_llama, f1_orca, f1_falcon))

# Conferma le modifiche e chiudi la connessione
conn.commit()
cursor.close()
conn.close()
