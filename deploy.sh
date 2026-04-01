#!/bin/bash
docker build --platform linux/amd64 -t dawa . && \
docker tag dawa europe-west9-docker.pkg.dev/gen-lang-client-0994342397/dawa-art/dawa:latest && \
docker push europe-west9-docker.pkg.dev/gen-lang-client-0994342397/dawa-art/dawa:latest && \
gcloud config set project gen-lang-client-0994342397 && \
gcloud run deploy dawa \
    --image=europe-west9-docker.pkg.dev/gen-lang-client-0994342397/dawa-art/dawa:latest \
    --region=europe-west9 \
    --memory 8Gi \
    --cpu=2 \
    --allow-unauthenticated && \
 gcloud run services update-traffic dawa --to-latest &&\
 gcloud config set run/region europe-west9
     