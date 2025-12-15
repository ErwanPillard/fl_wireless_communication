#!/bin/bash

N_CLIENTS=3

source env/bin/activate

# Chemin complet du projet
PROJECT_DIR="$(pwd)"

# Ouvre le serveur
osascript <<EOF
tell application "Terminal"
    do script "cd \"$PROJECT_DIR\" && source env/bin/activate && python -m server.server_base"
end tell
EOF

sleep 2

# Ouvre les clients
for i in $(seq 1 $N_CLIENTS)
do
osascript <<EOF
tell application "Terminal"
    do script "cd \"$PROJECT_DIR\" && source env/bin/activate && python -m client.client_base"
end tell
EOF
done
