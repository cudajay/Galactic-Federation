#!/bin/sh

sh /home/jmeow/git/Galactic-Federation/kubePilot/scripts/stop_deploys.sh
sh /home/jmeow/git/Galactic-Federation/kubePilot/scripts/reset_app.sh

kubectl -n rabbits apply -f /home/jmeow/git/Galactic-Federation/kubePilot/deployments/worker.yaml
kubectl -n rabbits apply -f /home/jmeow/git/Galactic-Federation/kubePilot/deployments/agg.yaml
sleep 5s
kubectl -n rabbits get pods
