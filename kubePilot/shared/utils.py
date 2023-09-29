import os
from datetime import date
import shutil
import json
import yaml
import optuna

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

def directory_manager(cfg, expn):
    today = str(date.today())
    dir_ = os.path.join("/app","data", "logs", f"{today}-{cfg['arch']}-{cfg['re']}")
    if os.path.exists(dir_) and expn == 0:
        shutil.rmtree(dir_)
        os.makedirs(dir_)
    dir_ = os.path.join(dir_, str(expn))
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    os.makedirs(dir_)
    file=open(os.path.join(dir_, "config.yaml"),"w")
    yaml.dump(cfg,file)
    file.close()
    json_object = json.dumps([], indent=4)
    with open(os.path.join(dir_,"training.json"), "w") as outfile:
        outfile.write(json_object)
    with open(os.path.join(dir_,"misc.json"), "w") as outfile:
        outfile.write(json_object)
    return dir_

def suggest_parameters(nsuggestions, pDicts):
    def objective(trail):
        ret = {}
        for d in pDicts:
            if d['type'] == 0:
                ret[d['param']] = trail.suggest_int(d['param'], *d['range'])  
            elif d['type'] == 1:
                ret[d['param']] = trail.suggest_float(d['param'], *d['range'])  
            else:
                ret[d['param']] = trail.suggest_categorical(d['param'], d['range'])
        return ret
    study = optuna.create_study()
    suggested_parameters = []
    for _ in range(nsuggestions):
        trial = study.ask()
        suggested_parameters.append(objective(trial))
    return suggested_parameters

def parse_init_config(cfg:dict, nsuggestions:int):
    param_list = []
    ret = []
    for k,v in cfg.items():
        if type(v) is dict:
            param_list.append({'param': k, 'type': v['type'], 'range':v['range']})
    suggested_parameters = suggest_parameters(nsuggestions, param_list)
    for d in suggested_parameters:
        tmp = cfg.copy()
        for k,v in d.items():
            tmp[k] = v
        ret.append(tmp)
    return ret


