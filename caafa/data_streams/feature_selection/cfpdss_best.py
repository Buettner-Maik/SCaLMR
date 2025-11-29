from .base import FeatureSelection
from ..constants import LABEL_DICT_KEY
import math

"""
4000
c6 -> 20%
c8 -> 50%
"""

"""
N(µ, s) -> normaldist with mean µ and stddev s
in x -> 1-x chance to flip the result
F(val, x) -> not of val with uniform random chance x
x ~ N(µ, s) -> x sampled from the distribution of N(µ, s)
x ~ d2 -> x sampled from a fair dice with 2 faces 
 c5 -> 'b'
!c5 -> 'a'
  y -> 'B'
 !y -> 'A'

cfpdss concepts:
    * n0, n1, n3 ~ N(0, 1)
    * n2 = 2 * (n1 + N(0, 1/8))
    * n4 = (n1 + n3) / 2 + N(0, 1/8)
    * c5, c6 ~ d2
    * c7 = c6 in 7/8
    * c8 = n3 < 0 in 7/8
    * c9 = c6 xor c8 in 7/8
    * chunk  1 - 10 label: [c7 and (n2 + n3 <  0.5)] or [!c7 and (n3 + n4 < -0.5)] in 31/32
    * chunk      11 label: [c7 and (n2 + n3 < -1  )] or [!c7 and (n3 + n4 <  0  )] in 31/32
    * chunk      12 label: [c6 and (n2 + n3 < -1  )] or [!c7 and (n3 + n4 <  0  )] in 31/32
    * chunk      13 label: [c5 and (n0 + n4 <  0.5)] or [!c5 and (n1      ≥  0.5)] in 31/32
    * chunk  1:     0 -   999
    * chunk  2:  1000 -  1999, linear  n0 (+2), n1 (+1) and n3 (-1)
    * chunk  3:  2000 -  2999, linear  c5 (80%), c6 (20%) and c8 (thr -1)
    * chunk  4:  3000 -  3999, linear  n2 (coef +5), n4 (wei .8, .2), c7 (flip .25) and c9 (flip .25)
    * chunk  5:  4000 -  4999, gradual n0 (-2), n1 (-1) and n3 (+1)
    * chunk  6:  5000 -  5999, gradual c5 (20%), c6 (80%) and c8 (thr +1)
    * chunk  7:  6000 -  6999, gradual n2 (coef -3), n4 (wei .2, .8), c7 (flip .375) and c9 (c5 xor c8)
    * chunk  8:  7000 -  7999, sudden  n0 (±0), n1 (±0) and n3 (±0)
    * chunk  9:  8000 -  8999, sudden  c5 (50%), c6 (50%) and c8 (thr ±0)
    * chunk 10:  9000 -  9999, sudden  n2 (coef +2), n4 (wei .5, .5), c7 (flip .125) and c9 (flip .125)
    * chunk 11: 10000 - 10999, linear  label
    * chunk 12: 11000 - 11999, gradual label
    * chunk 13: 12000 - 12999, sudden  label
"""

class Perfect_CFPDSS_Classifier():
    def __init__(self, first_inst_i=0):
        self.i = first_inst_i

    def learn_one(self, x, y):
        pass
    
    def predict_one(self, x):
        n0 = x["n0"]
        n1 = x["n1"]
        n2 = x["n2"]
        n3 = x["n3"]
        n4 = x["n4"]
        c5 = x["c5"] == 'b'
        c6 = x["c6"] == 'b'
        c7 = x["c7"] == 'b'
        if self.i < 10000:
            y = (c7 and (n2 + n3 < 0.5)) or (not c7 and (n3 + n4 < -0.5))
        elif self.i < 11000:
            y = (c7 and (n2 + n3 < -1)) or (not c7 and (n3 + n4 < 0))
        elif self.i < 12000:
            y = (c6 and (n2 + n3 < -1)) or (not c7 and (n3 + n4 < 0))
        else:
            y = (c5 and (n0 + n4 < 0.5)) or (not c5 and (n1 >= 0.5))
        self.i += 1

        return 'B' if y else 'A'

    def predict_proba_one(self, x):
        y_proba = {'A':0.0, 'B':0.0}
        y = self.predict_one(x)
        y_proba[y] = 1.0
        return y_proba
    
    def __str__(self):
        return f"PerfectCFPDSSClassifier {self.i}"

    def __repr__(self):
        return self.__str__()

class KBest_CFPDSS(FeatureSelection):
    def __init__(self, k, first_inst_i=0):
        self.i = first_inst_i
        self.k = k

    def transform_one(self, x):
        p = {}
        #x = x.copy()
        p[LABEL_DICT_KEY] = 100
        if self.i < 11000:
            p["c7"] = 19
            p["n3"] = 18
            p["n2"] = 17
            p["n4"] = 16
            p["n1"] = 5
            p["c6"] = 4
            p["c8"] = 3
            p["c9"] = 2
            p["c5"] = 0
            p["n0"] = 0
        elif self.i < 12000:
            p["c7"] = 19
            p["c6"] = 18
            p["n3"] = 17
            p["n2"] = 16
            p["n4"] = 15
            p["n1"] = 4
            p["c8"] = 3
            p["c9"] = 2
            p["c5"] = 0
            p["n0"] = 0
        else:
            p["c5"] = 19
            p["n1"] = 18
            p["n0"] = 17
            p["n4"] = 16
            p["n3"] = 5
            p["n2"] = 4
            p["c6"] = 0
            p["c7"] = 0
            p["c8"] = 0
            p["c9"] = 0
        self.i += 1

        return self._select_k_best({f: v for f, v in p.items() if f in x}, self.k)
        #return self._select_k_best(x, self.k)

class Perfect_CFPDSS_Imputer():
    def __init__(self, first_inst_i=0):
        self.i = first_inst_i
        self.t_time = 500

    def _get_simples(self, i):
        if i < 1000: 
            s = {"n0":0, "n1":0, "n2":0, "n3":0, "n4":0, 
                 "c5":'b', "c6":'b', "c7":'b', "c8":'b', "c9":'b'}
            return s
        elif i < 1500:
            s = {"n0":0, "n1":0, "n2":0, "n3":0, "n4":0, 
                 "c5":'b', "c6":'b', "c7":'b', "c8":'b', "c9":'a'}
            s["n0"] = (2 / self.t_time) * (i - 1000)
            s["n1"] = (1 / self.t_time) * (i - 1000)
            s["n3"] = (-1 / self.t_time) * (i - 1000)
            s["n2"] = s["n1"] * 2
            return s
        elif i < 2000:
            s = {"n0":2, "n1":1, "n2":2, "n3":-1, "n4":0, 
                 "c5":'b', "c6":'b', "c7":'b', "c8":'b', "c9":'a'}
            return s
        elif i < 3000:
            s = {"n0":2, "n1":1, "n2":2, "n3":-1, "n4":0, 
                 "c5":'b', "c6":'a', "c7":'b', "c8":'b', "c9":'a'}
            return s
        elif i < 3500:
            s = {"n0":2, "n1":1, "n2":2, "n3":-1, "n4":0, 
                 "c5":'b', "c6":'a', "c7":'b', "c8":'b', "c9":'a'}
            s["n2"] = 2 + (3 / self.t_time) * (i - 3000)
            return s
        elif i < 4000:
            s = {"n0":2, "n1":1, "n2":5, "n3":-1, "n4":0, 
                 "c5":'b', "c6":'a', "c7":'b', "c8":'b', "c9":'a'}
            return s
        elif i < 4500:
            s = {"n0":2, "n1":1, "n2":5, "n3":-1, "n4":0, 
                 "c5":'b', "c6":'a', "c7":'b', "c8":'a', "c9":'b'}
            s["n0"] = 2 + (-4 / self.t_time) * (i - 4000)
            s["n1"] = 1 + (-2 / self.t_time) * (i - 4000)
            s["n3"] = -1 + (2 / self.t_time) * (i - 4000)
            s["n2"] = s["n1"] * 5
            return s
        elif i < 5250:
            s = {"n0":-2, "n1":-1, "n2":-5, "n3":1, "n4":0, 
                 "c5":'b', "c6":'a', "c7":'b', "c8":'a', "c9":'b'}
            return s
        elif i < 6000:
            s = {"n0":-2, "n1":-1, "n2":-5, "n3":1, "n4":0, 
                 "c5":'a', "c6":'b', "c7":'b', "c8":'a', "c9":'b'}
            return s
        elif i < 6500:
            s = {"n0":-2, "n1":-1, "n2":-5, "n3":1, "n4":0, 
                 "c5":'a', "c6":'b', "c7":'b', "c8":'a', "c9":'b'}
            s["n2"] = 5 + (-8 / self.t_time) * (i - 6000)
            return s
        elif i < 7000:
            s = {"n0":-2, "n1":-1, "n2":3, "n3":1, "n4":0, 
                 "c5":'a', "c6":'b', "c7":'b', "c8":'a', "c9":'b'}
            return s
        elif i < 8000:
            s = {"n0":0, "n1":0, "n2":0, "n3":0, "n4":0, 
                 "c5":'a', "c6":'b', "c7":'b', "c8":'b', "c9":'b'}
            return s
        elif i < 9000:
            s = {"n0":0, "n1":0, "n2":0, "n3":0, "n4":0, 
                 "c5":'b', "c6":'b', "c7":'b', "c8":'b', "c9":'b'}
            return s
        else:
            s = {"n0":0, "n1":0, "n2":0, "n3":0, "n4":0, 
                 "c5":'b', "c6":'b', "c7":'b', "c8":'b', "c9":'b'}
            return s

    def learn_one(self, x, y):
        pass

    def _get_n1_from_n2(self, i, n2):
        return n2 / 2

    def _get_n1_from_n3_n4(self, i, n3, n4):
        return 2 * n4 - n3

    def _get_n2_from_n1(self, i, n1):
        return n1 * 2

    def _get_n3_from_n1_n4(self, i, n1, n4):
        return 2 * n4 - n1

    def _get_n3_from_c8(self, i, c8):
        return math.sqrt(2 / math.pi) * (-1 if c8 else 1)  #truncated stddev

    def _get_n4_from_n1_n3(self, i, n1, n3):
        return (n1 + n3) / 2

    def _get_c6_from_c7(self, i, c7):
        pass

    def _get_c6_from_c8_c9(self, i, c8, c9):
        pass

    def _get_c6_from_c9(self, i, c9):
        pass

    def _get_c7_from_c6(self, i, c6):
        pass

    def _get_c8_from_n3(self, i, n3):
        pass

    def _get_c8_from_c6_c9(self, i, c6, c9):
        pass

    def _get_c8_from_c9(self, i, c9):
        pass

    def _get_c9_from_c6_c8(self, i, c6, c8):
        pass

    def _get_c9_from_c6(self, i, c6):
        pass

    def _get_c9_from_c8(self, i, c8):
        pass


    def transform_one(self, x):
        x_imp = x.copy()
        simples = self._get_simples(self.i)
        todos = [f for f, v in x.items() if v is None]


        n0 = x["n0"] is not None
        n1 = x["n1"] is not None
        n2 = x["n2"] is not None
        n3 = x["n3"] is not None
        n4 = x["n4"] is not None
        c5 = x["c5"] is not None
        c6 = x["c6"] is not None
        c7 = x["c7"] is not None
        c8 = x["c8"] is not None
        c9 = x["c9"] is not None



        if not n0:
            x_imp["n0"] = simples["n0"]

        if not n1:
            if not n2:
                x_imp["n1"] = simples["n1"]
            else:
                if self.i < 3000:
                    x_imp["n1"] = x_imp["n2"] / 2
                elif self.i <= 3500:
                    x_imp["n1"] = x_imp["n2"] / (2 + (3 / self.t_time) * (self.i - 3000))
                elif self.i < 6000:
                    x_imp["n1"] = x_imp["n2"] / 5
                elif self.i <= 6500:
                    x_imp["n1"] = x_imp["n2"] / (5 + (-8 / self.t_time) * (self.i - 3000))
                elif self.i < 9000:
                    x_imp["n1"] = x_imp["n2"] / -3
                else:
                    x_imp["n1"] = x_imp["n2"] / 2

        if not n2:
            if not n1:
                x_imp["n2"] = simples["n2"]
            else:
                if self.i < 3000:
                    x_imp["n2"] = x_imp["n1"] * 2
                elif self.i <= 3500:
                    x_imp["n2"] = x_imp["n1"] * (2 + (3 / self.t_time) * (self.i - 3000))
                elif self.i < 6000:
                    x_imp["n2"] = x_imp["n1"] * 5
                elif self.i <= 6500:
                    x_imp["n2"] = x_imp["n1"] * (5 + (-8 / self.t_time) * (self.i - 3000))
                elif self.i < 9000:
                    x_imp["n2"] = x_imp["n1"] * -3
                else:
                    x_imp["n2"] = x_imp["n1"] * 2
        
        if not n3:
            pass
            #x["n3"] =
            


        if self.i < 1000:
            x["n0"]

        n0 = x["n0"]
        n1 = x["n1"]
        n2 = x["n2"]
        n3 = x["n3"]
        n4 = x["n4"]
        c5 = x["c5"] == 'b'
        c6 = x["c6"] == 'b'
        c7 = x["c7"] == 'b'
        if self.i < 10000:
            y = (c7 and (n2 + n3 < 0.5)) or (not c7 and (n3 + n4 < -0.5))
        elif self.i < 11000:
            y = (c7 and (n2 + n3 < -1)) or (not c7 and (n3 + n4 < 0))
        elif self.i < 12000:
            y = (c6 and (n2 + n3 < -1)) or (not c7 and (n3 + n4 < 0))
        else:
            y = (c5 and (n0 + n4 < 0.5)) or (not c5 and (n1 >= 0.5))
        self.i += 1

        return 'B' if y else 'A'

    def predict_proba_one(self, x):
        y_proba = {'A':0.0, 'B':0.0}
        y = self.predict_one(x)
        y_proba[y] = 1.0
        return y_proba
    
    def __str__(self):
        return f"PerfectCFPDSSClassifier {self.i}"

    def __repr__(self):
        return self.__str__()