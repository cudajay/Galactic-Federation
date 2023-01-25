#!/bin/sh
 kubectl -n rabbits exec rabbitmq-0 -- rabbitmqctl stop_app
 kubectl -n rabbits exec rabbitmq-0 -- rabbitmqctl reset
 kubectl -n rabbits exec rabbitmq-0 -- rabbitmqctl start_app