#!/bin/sh

eval $(minikube docker-env)
cd publisher
docker build -t pub3 .
cd ../consumer
docker build -t cons1 .
kubectl -n rabbits delete deployment cons1
kubectl -n rabbits delete deployment pub3
cd ../
kubectl -n rabbits apply -f consumer/deployment.yaml
kubectl -n rabbits apply -f publisher/deployment.yaml
sleep 10s
kubectl -n rabbits get pods
