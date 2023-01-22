#!/bin/sh

sh scripts/stop_deploys.sh
sh scripts/reset_app.sh

kubectl -n rabbits apply -f deployments/agg.yaml
kubectl -n rabbits apply -f deployments/worker.yaml
sleep 3s
kubectl -n rabbits get pods