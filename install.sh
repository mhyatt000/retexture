#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "You are using MacOS."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew install blender python

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "You are using Linux."
    apt-get update
    apt-get install blender python3

else
  echo "Unknown OS."
fi

pip install -r requirements.txt
