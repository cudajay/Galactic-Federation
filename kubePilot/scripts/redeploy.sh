#!/bin/sh

sh scripts/stop_deploys.sh
sh scripts/reset_app.sh

kubectl -n rabbits apply -f deployments/worker.yaml
kubectl -n rabbits apply -f deployments/agg.yaml
sleep 5s
kubectl -n rabbits get pods