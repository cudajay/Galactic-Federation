#!/bin/sh

eval $(minikube docker-env)
kubectl -n rabbits delete service/test-viz
kubectl -n rabbits delete deployment.apps/viz

cd Viz
docker build -t viz .
cd ..
kubectl -n rabbits create -f deployments/viz.yaml
minikube -n rabbits service test-viz
sleep 3s
kubectl -n rabbits port-forward service/test-viz 8000:8000

