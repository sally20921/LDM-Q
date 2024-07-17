import os

quant_type = os.environ.get("QUANT_TYPE", "LSQ")

if quant_type == "LSQ" or quant_type == 'lsq':
    from .lsq import QuantOps as Q



