from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import sys

# Add src to path so DAGs can import ml_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.model import train_model_v2, evaluate_model_v2, promote_model_v2

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="ml_training_pipeline_v2",
    default_args=default_args,
    description="HW3: train -> evaluate -> promote (versioned, S3)",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    # Airflow provides run timestamp in ts_nodash like 20260326T012345
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model_v2,
        op_kwargs={
            "model_dir": "models",
            "model_version": "{{ ts_nodash }}",
        },
    )

    eval_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model_v2,
        op_kwargs={
            "model_dir": "models",
            "model_version": "{{ ts_nodash }}",
        },
    )

    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model_v2,
        op_kwargs={
            "model_dir": "models",
            "model_version": "{{ ts_nodash }}",
            "threshold": 0.94,
            
        },
    )

    train_task >> eval_task >> promote_task