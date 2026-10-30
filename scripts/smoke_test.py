import sys
sys.path.insert(0, "src")
import yaml, torch
from friday.model.config import FridayConfig
from friday.model.model import FridayModel
for cfgp in ["configs/nano.yaml","configs/flash.yaml","configs/base.yaml"]:
    raw = yaml.safe_load(open(cfgp))
    cfg = FridayConfig(**raw["model"])
    m = FridayModel(cfg)
    print(cfgp, "params:", m.num_params(), "size_int4_MB:", round(m.num_params()/2/1e6,1))
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = m(x)
    print("  forward ok:", tuple(y.shape))
