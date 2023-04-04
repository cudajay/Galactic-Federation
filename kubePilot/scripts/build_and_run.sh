#!/bin/sh
eval $(minikube docker-env)

if [ "$1" = "1" ]; then
echo "Debugging consumer";
docker build -f debugger/dockerfile --no-cache -t cons1 .
docker build -f publisher/dockerfile --no-cache -t pub3 .
elif [ "$1" = "2" ]; then
echo "Debugging publisher";
docker build -f consumer/dockerfile --no-cache -t cons1 .
docker build -f debugger/dockerfile --no-cache -t pub3 .
elif [ "$1" = "0" ]; then
echo "Normal Operation";
docker build -f consumer/dockerfile --no-cache -t cons1 .
docker build -f publisher/dockerfile --no-cache -t pub3 .
fi

sh scripts/redeploy.sh
