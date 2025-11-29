Docker Instructions:

## Docker Instructions

### Build the container:

docker build -t model-server:v1 .

### Run the API server:

docker run -p 8000:8000 model-server:v1

### Test the app:

curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'
curl http://localhost:8000/metrics

### Run batch inference in container:

docker exec -it <container_id> python batch_infer.py data/input.csv data/predictions.csv

### Screenshots

See the screenshots folder for:

    * metrics screenshot