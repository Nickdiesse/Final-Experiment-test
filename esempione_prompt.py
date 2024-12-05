from gpt4all import GPT4All
import json
import psycopg2
import time

# Create a connection to your db
def connect_to_db():
    return psycopg2.connect(
        host="localhost",  
        database="Esempionefinale",  
        user="postgres",  
        password="nicola"  
    )

# 1. Load the models
model_orca = GPT4All(r"C:\Users\nicol\gpt4all\resources\orca-2-7b.Q4_0.gguf")
model_llama = GPT4All(r"C:\Users\nicol\gpt4all\resources\Meta-Llama-3-8B-Instruct.Q4_0.gguf")
model_falcon = GPT4All(r"C:\Users\nicol\gpt4all\resources\gpt4all-falcon-newbpe-q4_0.gguf")

# 2. Load the Json file
with open(r"C:\Users\nicol\Desktop\progetto_tesi\Esempione_finale\esempione_def.json") as f:
    esempione_data = json.load(f)

# 3. Function to ask question to our models using gpt4all
def ask_question_gpt4all(model, question):
    prompt = f"Question: {question}\nAnswer:"
    response = model.generate(prompt)
    return response

# 4. Funzione to save results in database
def save_to_db(cur, question,ground_truth, answer_orca, answer_llama, answer_falcon, time_orca, time_llama, time_falcon):
    cur.execute("""
        INSERT INTO esempione_prompt_db (question,ground_truth, answer_orca, answer_llama, answer_falcon, time_orca, time_llama, time_falcon)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """, (question, ground_truth, answer_orca, answer_llama, answer_falcon, time_orca, time_llama, time_falcon))

# 5. Connection to the database
conn = connect_to_db()
cur = conn.cursor()

# Iterate through the JSON data and process each question
for item in esempione_data:
    question = item.get("question")
    ground_truth = item.get("ground_truth", "")

    # Measure response time for each model
    start_orca = time.time()
    answer_orca = ask_question_gpt4all(model_orca, question)
    time_orca = time.time() - start_orca

    start_llama = time.time()
    answer_llama = ask_question_gpt4all(model_llama, question)
    time_llama = time.time() - start_llama

    start_falcon = time.time()
    answer_falcon = ask_question_gpt4all(model_falcon, question)
    time_falcon = time.time() - start_falcon

    # Save results to the database
    save_to_db(cur, question, ground_truth, answer_orca, answer_llama, answer_falcon, time_orca, time_llama, time_falcon)

# Commit changes and close the connection
conn.commit()
cur.close()
conn.close()