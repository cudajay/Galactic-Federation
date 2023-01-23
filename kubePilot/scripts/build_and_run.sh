#!/bin/sh

eval $(minikube docker-env)
docker build -f consumer/dockerfile --no-cache -t cons1 .
docker build -f publisher/dockerfile --no-cache -t pub3 .

sh scripts/redeploy.sh
