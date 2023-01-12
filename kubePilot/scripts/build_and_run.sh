#!/bin/sh

eval $(minikube docker-env)
docker build -f consumer/dockerfile -t cons1 .
docker build -f publisher/dockerfile -t pub3 .

sh scripts/redeploy.sh
