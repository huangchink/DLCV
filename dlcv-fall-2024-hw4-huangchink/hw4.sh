#!/bin/bash
python3 gaussian-splatting/render.py -m ./ --source_path $1 --output_dir $2

# TODO - run your inference Python3 code