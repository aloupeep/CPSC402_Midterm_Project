#!/bin/bash
# Upload the packaged archive to gofile.io (anonymous guest upload) and print the download page link.
set -e
F=${1:-/home/worker/lab/mixed_precision_writeup_results.zip}
ls -la "$F"
sha256sum "$F" | tee "$F.sha256"
srv=$(curl -s https://api.gofile.io/servers | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data']['servers'][0]['name'])")
echo "upload server: $srv"
resp=$(curl -s -F "file=@$F" "https://$srv.gofile.io/contents/uploadfile")
echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print('status', d.get('status')); print('DOWNLOAD PAGE:', d['data']['downloadPage']); print('name', d['data']['name'], 'md5', d['data']['md5'])"
