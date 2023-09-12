#!/bin/sh
cd /home/jmeow/git/Galactic-Federation
kubectl create ns rabbits
kubectl apply -n rabbits -f kubePilot/rabbitmqConfig/rbac.yaml
kubectl apply -n rabbits -f kubePilot/rabbitmqConfig/configmap.yaml
kubectl apply -n rabbits -f kubePilot/rabbitmqConfig/secrets.yaml
kubectl apply -n rabbits -f kubePilot/rabbitmqConfig/stateful.yaml 