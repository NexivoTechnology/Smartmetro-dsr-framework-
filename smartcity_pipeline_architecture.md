# SmartMetro IoT Data Pipeline Architecture
**Design Science Research Project | Version 1.0**

---

## Overview

This document specifies the end-to-end data pipeline for the SmartMetro smart city IoT platform. The architecture follows a **Lambda Architecture** pattern supporting both real-time stream processing and batch analytics, with a dedicated AI/LLM serving layer.

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: INGESTION                                                  │
│  IoT Sensors → MQTT Broker → Apache Kafka Topics                    │
│  Topics: smartcity.traffic | smartcity.energy | smartcity.air       │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│  LAYER 2: STREAM PROCESSING (Speed Layer)                            │
│  Apache Kafka Streams / PySpark Streaming                            │
│  - Schema validation & null checks                                   │
│  - Real-time anomaly detection (3-sigma rule)                        │
│  - Enrichment: weather API join, zone metadata join                  │
│  Output: Validated records → Redis cache (hot data, 1hr TTL)         │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│  LAYER 3: BATCH PROCESSING (Batch Layer)                             │
│  Apache Spark on Databricks / EMR                                    │
│  - Historical reprocessing of raw sensor logs                        │
│  - Feature engineering for ML models                                 │
│  - Data quality scoring                                               │
│  Output: Delta Lake Parquet partitioned by domain/date               │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│  LAYER 4: STORAGE (Data Lakehouse)                                   │
│  ├── Raw Zone:       S3://smartmetro/raw/{domain}/{date}/            │
│  ├── Processed Zone: S3://smartmetro/processed/{domain}/{date}/      │
│  └── Curated Zone:   S3://smartmetro/curated/unified_iot_master/     │
│  Format: Delta Lake (Parquet + transaction log)                      │
│  Catalog: AWS Glue / Apache Hive Metastore                           │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
        ┌─────────────┼──────────────┐
        ▼             ▼              ▼
┌───────────┐  ┌────────────┐  ┌────────────────────────┐
│ ANALYTICS │  │  ML LAYER  │  │   AI / RAG LAYER        │
│  SERVING  │  │            │  │                          │
│ DuckDB /  │  │ Scikit-    │  │ ChromaDB Vector Store    │
│ Redshift  │  │ learn      │  │ LLaMA-3 Embeddings       │
│ Tableau / │  │ LightGBM   │  │ LLaMA-3-8B Generator     │
│ Superset  │  │ Prophet    │  │ FastAPI RAG Endpoint     │
└───────────┘  └────────────┘  └────────────────────────┘
```

---

## ETL Pseudocode

### Step 1: Ingest from Kafka

```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'smartcity.traffic', 'smartcity.energy', 'smartcity.air',
    bootstrap_servers=['kafka-broker:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

for msg in consumer:
    record = msg.value
    domain = msg.topic.split('.')[-1]
    process_record(record, domain)
```

### Step 2: Validate & Clean

```python
def validate_record(record: dict, domain: str) -> dict:
    # 1. Schema enforcement
    required_fields = ['sensor_id', 'timestamp', 'zone']
    assert all(f in record for f in required_fields), "Missing required fields"

    # 2. Timestamp parsing
    record['timestamp'] = pd.to_datetime(record['timestamp'])

    # 3. Domain-specific range checks
    if domain == 'traffic':
        record['vehicle_count'] = max(0, min(600, record.get('vehicle_count', 0)))
        record['congestion_index'] = max(0.0, min(1.0, record.get('congestion_index', 0)))
    elif domain == 'air_quality':
        record['aqi_score'] = max(0, min(500, record.get('aqi_score', 0)))
    elif domain == 'energy':
        record['renewable_pct'] = max(0.0, min(100.0, record.get('renewable_pct', 0)))

    # 4. Data quality scoring
    null_count = sum(1 for v in record.values() if v is None or v == '')
    record['data_quality_score'] = round(1 - (null_count / len(record)), 2)

    # 5. Anomaly detection (3-sigma)
    record['anomaly_flag'] = detect_anomaly(record, domain)
    return record
```

### Step 3: Anomaly Detection

```python
from collections import deque
import statistics

sensor_history = {}  # {sensor_id: deque of recent values}

def detect_anomaly(record: dict, domain: str) -> int:
    sid = record['sensor_id']
    
    value_map = {
        'traffic': record.get('vehicle_count'),
        'energy': record.get('energy_kwh'),
        'air_quality': record.get('aqi_score')
    }
    val = value_map.get(domain)
    if val is None:
        return 0
    
    if sid not in sensor_history:
        sensor_history[sid] = deque(maxlen=100)
    
    history = sensor_history[sid]
    if len(history) >= 30:
        mean = statistics.mean(history)
        stdev = statistics.stdev(history)
        if stdev > 0 and abs(val - mean) > 3 * stdev:
            return 1
    
    history.append(val)
    return 0
```

### Step 4: Write to Delta Lake

```python
from delta import DeltaTable
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SmartMetroETL") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .getOrCreate()

def write_to_lakehouse(records: list, domain: str):
    df = spark.createDataFrame(records)
    df.write \
        .format("delta") \
        .mode("append") \
        .partitionBy("domain", "zone") \
        .save(f"s3://smartmetro/processed/{domain}/")
    print(f"Written {df.count()} records to Delta Lake for domain: {domain}")
```

---

## RAG Pipeline Implementation

### Embedding & Indexing

```python
import chromadb
from llama_index.embeddings import LlamaEmbedding
import pandas as pd

# Load RAG chunks CSV
chunks_df = pd.read_csv("smartcity_rag_chunks.csv")

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./smartcity_vectorstore")

# Create domain-partitioned collections
collections = {
    domain: client.get_or_create_collection(f"smartcity_idx_{domain}")
    for domain in ["traffic", "energy", "air_quality"]
}

# Embed and index all chunks
embed_model = LlamaEmbedding(model_name="meta-llama/Meta-Llama-3-8B")

for _, row in chunks_df.iterrows():
    embedding = embed_model.get_text_embedding(row['text_chunk'])
    collections[row['domain']].add(
        ids=[row['chunk_id']],
        embeddings=[embedding],
        documents=[row['text_chunk']],
        metadatas=[{
            "zone": row['zone'],
            "document_type": row['document_type'],
            "domain": row['domain'],
            "source_doc_id": row['source_doc_id']
        }]
    )
print("All chunks embedded and indexed.")
```

### RAG Query & Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load LLaMA-3-8B
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

SYSTEM_PROMPT = """You are SmartMetro AI, an intelligent urban analytics assistant.
Answer using ONLY the retrieved context chunks. Cite chunk_id sources."""

def query_smartcity(question: str, domain: str = None, top_k: int = 5) -> str:
    # 1. Embed query
    query_embedding = embed_model.get_text_embedding(question)
    
    # 2. Retrieve relevant chunks
    target_collections = [collections[domain]] if domain else list(collections.values())
    all_results = []
    for col in target_collections:
        results = col.query(query_embeddings=[query_embedding], n_results=top_k)
        all_results.extend(zip(results['ids'][0], results['documents'][0]))
    
    # Sort by relevance, take top_k
    context = "\n\n".join([f"[{cid}]: {doc}" for cid, doc in all_results[:top_k]])
    
    # 3. Generate with LLaMA
    prompt = f"<|system|>{SYSTEM_PROMPT}<|user|>Context:\n{context}\n\nQuestion: {question}<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
response = query_smartcity(
    "What anomalies were detected in Zone-D-Industrial for air quality last month?",
    domain="air_quality"
)
print(response)
```

---

## Airflow DAG (Orchestration)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'smartmetro-team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
}

with DAG(
    'smartmetro_iot_pipeline',
    default_args=default_args,
    schedule_interval='*/30 * * * *',  # Every 30 minutes
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['smartcity', 'iot', 'etl']
) as dag:

    ingest = PythonOperator(task_id='ingest_kafka', python_callable=ingest_from_kafka)
    validate = PythonOperator(task_id='validate_clean', python_callable=validate_and_clean)
    enrich = PythonOperator(task_id='enrich_records', python_callable=enrich_with_metadata)
    write_lake = PythonOperator(task_id='write_delta_lake', python_callable=write_to_lakehouse)
    update_rag = PythonOperator(task_id='update_rag_index', python_callable=update_vector_index)

    ingest >> validate >> enrich >> write_lake >> update_rag
```

---

## Data Quality Checks

| Check | Rule | Action on Failure |
|-------|------|-------------------|
| Null rate | < 5% nulls per batch | Flag batch, alert ops team |
| Schema drift | All expected columns present | Dead-letter queue |
| Timestamp gap | No gap > 6 hours per sensor | Impute or mark missing |
| Value range | Domain-specific min/max bounds | Clamp + anomaly_flag = 1 |
| Duplicate records | record_id uniqueness | Deduplicate on ingest |
| Battery threshold | sensor_battery_pct >= 15% | Dispatch maintenance alert |

---

## Technology Stack Summary

| Component | Technology | Justification |
|-----------|------------|---------------|
| Message Queue | Apache Kafka | High-throughput, durable, IoT-proven |
| Stream Processing | PySpark Streaming | Unified batch/stream API |
| Orchestration | Apache Airflow | Rich scheduling, retry, monitoring |
| Data Lake | Delta Lake on S3 | ACID transactions + Parquet efficiency |
| Analytics Query | DuckDB (local) / Redshift (cloud) | Fast SQL on Parquet |
| Vector Store | ChromaDB | Lightweight, LlamaIndex-native |
| LLM | LLaMA-3-8B-Instruct | Open source, self-hostable, no API costs |
| Embedding | LLaMA-3-Embed | Domain-consistent embeddings |
| API Layer | FastAPI | Async Python, OpenAPI docs |
| Visualization | Apache Superset | Open source BI, SQL-native |
