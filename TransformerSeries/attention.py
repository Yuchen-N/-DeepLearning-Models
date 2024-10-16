import copy

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def clones(module, N):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = layer
        self.norm = LayerNorm(layer.size)
