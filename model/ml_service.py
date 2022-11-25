from app.detect import parse_opt
from app.detect import main
from dotenv import load_dotenv
import json
import yaml
import os

load_dotenv()

# Load env variables
database = os.getenv('DATABASE')
project = os.getenv('PROJECT')
weights = os.getenv('WEIGTHS')
save_txt = os.getenv('SAVE_TXT')
source = "/home/src/app/data/images/zidane.jpg"


# Get the preset values from the parse function
args = parse_opt()

# Change values
args.save_conf = True
args.save_txt = True
args.project = project
# args.weights = weights
args.source = source
args.name = "zidane"

run = main(args)

