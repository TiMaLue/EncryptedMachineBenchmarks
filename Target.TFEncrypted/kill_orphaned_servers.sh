#!/usr/bin/env bash
kill -9 $(lsof -t -i:10000)
kill -9 $(lsof -t -i:10001)
kill -9 $(lsof -t -i:10002)
kill -9 $(lsof -t -i:10003)
kill -9 $(lsof -t -i:10004)
exit 0