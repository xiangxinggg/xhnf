# --*-- coding:utf-8 --*--
from xhnf import XHNF
import numpy as np
seed = 7
np.random.seed(seed)

def main():
    print("Start XHNF Train.")
    xhnf = XHNF()
    xhnf.do_all()
    print("XHNF Train Done.")

if __name__ == '__main__':
    main()
