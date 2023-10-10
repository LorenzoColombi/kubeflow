# Deploy MLflow su K8s (a fianco a Kubeflow)


- Applicare tutti i file yaml
- Modificare la dashboard aggiungendo il link a MLflow
    
    ```bash
    kubectl edit cm centraldashboard-config -n kubeflow
    # add this under the other menu items
    {
    “type”: “item”,
    “link”: “/mlflow/”,
    “text”: “MlFlow”,
    “icon”: “icons:cached”
    }
    ```

## todo
Condigurare MLflow per usare Minio / mysql come storage