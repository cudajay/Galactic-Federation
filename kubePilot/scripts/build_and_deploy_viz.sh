#!/bin/sh

eval $(minikube docker-env)
cd Viz
docker build -t viz .
kubectl -n rabbits delete service/test-viz
kubectl -n rabbits delete deployment.apps/viz
cd ..
kubectl -n rabbits create -f deployments/viz.yaml
minikube -n rabbits service test-viz