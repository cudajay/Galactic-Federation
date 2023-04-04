import os
from datetime import date
import shutil
import json

class IIterable:
    def __init__(self, x, y, chunk_size) -> None:
        self.x = x
        self.y = y
        self.n = self.x.shape[0]
        self.cs = chunk_size
        self.pos = 0

    def __next__(self):
        x = self.x[self.pos:self.pos+self.cs, :, :]
        y = self.y[self.pos:self.pos+self.cs, :]

        if self.pos+self.cs + self.cs > self.n:
            self.pos = 0
        else:
            self.pos += self.cs
        return (x,y)

class Rule:
    def __init__(self,rule_handler, action_handler) -> None:
        self.rule_handler = rule_handler
        self.action_handler = action_handler
    def execute(self, *args,**kwargs):
        self.action_handler(*self.rule_handler(*args, **kwargs))

def directory_manager(cfg):
    today = str(date.today())
    dir_ = os.path.join("data", "logs", f"{today}-{cfg['re']}")
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    os.makedirs(dir_)
    json_object = json.dumps([], indent=4)
    with open(os.path.join(dir_,"training.json"), "w") as outfile:
        outfile.write(json_object)
    with open(os.path.join(dir_,"misc.json"), "w") as outfile:
        outfile.write(json_object)
    return dir_
