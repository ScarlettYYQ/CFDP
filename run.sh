#!/bin/bash

python CFDP/generate_cot_wrong.py --key REMOVED --base_url REMOVED
python CFDP/generate_input.py
python CFDP/demo.py --key REMOVED --base_url REMOVED --type ambig
