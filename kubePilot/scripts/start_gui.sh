#!/bin/sh

kubectl -n rabbits port-forward rabbitmq-0 8080:15672