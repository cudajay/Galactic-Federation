#!/bin/sh

kubectl -n rabbits delete deployment cons1
kubectl -n rabbits delete deployment pub3

kubectl -n rabbits apply -f deployments/agg.yaml
kubectl -n rabbits apply -f deployments/worker.yaml
sleep 3s
kubectl -n rabbits get pods