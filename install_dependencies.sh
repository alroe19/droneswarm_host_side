#!/bin/bash
# THE ABOVE LINE IS REQUIRED FOR THE SCRIPT TO BE EXECUTABLE AS A BASH SCRIPT - DO NOT REMOVE IT!
# Make this file executable with: chmod +x install_dependencies.sh (just has to be done once)
# Run this script with: ./install_dependencies.sh

set -e  # Stop the script if any command fails

apt install python3-pipx -y

apt update -y
apt full-upgrade -y
apt install -y python3-picamera2 --no-install-recommends
apt install imx500-all -y

python3 -m pipx install -r requirements.txt
