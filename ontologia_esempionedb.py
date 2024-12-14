from gpt4all import GPT4All
import psycopg2
import json
import time
from rdflib import Graph



# Impostare la connessione al database
def connect_to_db():
    return psycopg2.connect(
        host="localhost",  
        database="",  
        user="",  
        password=""  
    )

# 1. Carica i modelli locali
model_orca = GPT4All(r"C:\Users\nicol\gpt4all\resources\orca-2-7b.Q4_0.gguf")
model_llama = GPT4All(r"C:\Users\nicol\gpt4all\resources\Meta-Llama-3-8B-Instruct.Q4_0.gguf")
model_falcon = GPT4All(r"C:\Users\nicol\gpt4all\resources\gpt4all-falcon-newbpe-q4_0.gguf")

# Carica il file JSON con le domande
with open(r"C:\Users\nicol\Desktop\progetto_tesi\Esempione_finale\esempione_def.json") as f:
    questions = json.load(f)

# Load RDF data and set up retrieval function
def load_rdf_data(rdf_file_path):
    graph = Graph()
    graph.parse(rdf_file_path, format="xml")  # Adjust format if necessary
    return graph


def retrieve_context(graph, question):
    # Customize the SPARQL query to extract relevant info based on the question
    # Example: find triples that match certain keywords
    query = """
    SELECT ?subject ?predicate ?object
    WHERE {
        ?subject ?predicate ?object .
        FILTER(CONTAINS(LCASE(STR(?object)), LCASE("{question_text}")))
    }
    """.replace("{question_text}", question)
    results = graph.query(query)
    context = " ".join([str(row.object) for row in results])
    return context


# Connect to RDF file
rdf_graph = load_rdf_data(r"C:\Users\nicol\Desktop\progetto_tesi\Esempione_finale\Ontology\ontologia_completa.rdf")


# Define function to query models and save results
def process_questions(questions, graph):
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # SQL command to insert data
    insert_query = """
    INSERT INTO esempione_ontologia_db (id, question, ground_truth, orca_answer, llama_answer, falcon_answer, 
                           orca_time, llama_time, falcon_time)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    for item in questions:
        question_id = item["id"]
        question_text = item["question"]
        ground_truth = item["ground_truth"]
        
        # Retrieve context from RDF
        context = retrieve_context(graph, question_text)
        input_text = f"Question: {question_text}\nContext: {context}"
        
        # Query each model and record response times
        start_time = time.time()
        answer_orca = model_orca.generate(input_text)
        time_response_orca = time.time() - start_time

        start_time = time.time()
        answer_llama = model_llama.generate(input_text)
        time_response_llama = time.time() - start_time

        start_time = time.time()
        answer_falcon = model_falcon.generate(input_text)
        time_response_falcon = time.time() - start_time
        
        # Save results to database
        cursor.execute(insert_query, (question_id, question_text, ground_truth, answer_orca, 
                                      answer_llama, answer_falcon, time_response_orca, 
                                      time_response_llama, time_response_falcon))
        
    
    conn.commit()
    cursor.close()
    conn.close()

# Process all questions
process_questions(questions, rdf_graph)
print(f"Saved response and times for question ID {id} to the database.")
print("Processing complete, results saved to database.")
