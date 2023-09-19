#https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91

class SaveOutput:
    def __init__(self):
        self.outputs = None

    def __call__(self, module, module_in, module_out):
        self.outputs = module_out
