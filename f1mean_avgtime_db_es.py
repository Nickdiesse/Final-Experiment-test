import psycopg2

# Connessione al database PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="",
    user="",
    password=""
)
cursor = conn.cursor()

# Calcola la media dell'F1-score per ogni modello
cursor.execute("""
    SELECT 
        AVG(f1_llama) AS avg_f1_llama,
        AVG(f1_orca) AS avg_f1_orca,
        AVG(f1_falcon) AS avg_f1_falcon
    FROM f1score_csv_es;
""")
f1_averages = cursor.fetchone()

# Calcola la media del tempo di risposta per ogni modello
cursor.execute("""
    SELECT 
        AVG(response_time_llama) / 1000.0 AS avg_response_time_llama_sec,
        AVG(response_time_orca) / 1000.0 AS avg_response_time_orca_sec,
        AVG(response_time_falcon) / 1000.0 AS avg_response_time_falcon_sec
    FROM ragdb_esempione;
""")
response_time_averages = cursor.fetchone()

# Stampa le medie su terminale
print("F1-score mean for each model :")
print(f"Llama: {f1_averages[0]:.2f}")
print(f"Orca: {f1_averages[1]:.2f}")
print(f"Falcon: {f1_averages[2]:.2f}")

print("\nAverage time response for each model (in seconds):")
print(f"Llama: {response_time_averages[0]:.2f} s")
print(f"Orca: {response_time_averages[1]:.2f} s")
print(f"Falcon: {response_time_averages[2]:.2f} s")

# Chiudi la connessione
cursor.close()
conn.close()
