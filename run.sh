#!/bin/bash

python CFDP/CoT_Wrong.py --key YOURKEY --base_url URL
python CFDP/generate.py
python CFDP/demo.py --key YOURKEY --base_url URL --type ambig
