#!/bin/bash

python CFDP/generate_cot_wrong.py --key YOURKEY --base_url URL
python CFDP/generate_input.py
python CFDP/demo.py --key YOURKEY --base_url URL --type ambig
