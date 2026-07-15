#!/usr/bin/env bash

celery -A celery -A config.celery call atrag.tasks.index.add_index_for_local_document --args='["11"]'