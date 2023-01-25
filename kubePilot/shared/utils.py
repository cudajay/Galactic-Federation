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
