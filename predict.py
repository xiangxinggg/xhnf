# --*-- coding:utf-8 --*--
from xhnf import XHNF
import numpy as np
seed = 7
np.random.seed(seed)

def main():
    print("Start XHNF Predict.")
    xhnf = XHNF()
    xhnf.do_predict()
    print("XHNF Predict Done.")

if __name__ == '__main__':
    main()
