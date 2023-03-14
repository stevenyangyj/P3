import numpy as np

class TestFun(object):

    def __init__(self, name):
        
        self.name = name

    def step(self, vec):

        if self.name == "Schaffer":
            assert vec.size == 1
            f1 = vec**2
            f2 = (vec - 2)**2

        elif self.name == "ZDT3":
            assert vec.size == 2
            f1=vec[0]
            h=1+vec[1]
            f2=h*(1-(f1/h)**2-(f1/h)*np.sin(10*np.pi*f1))
        elif self.name == "ZDT1":
            assert vec.size == 2
            vec = np.clip(vec, 0, 1)
            f1 = vec[0]  # objective 1
            g = 1 + 9 * np.sum(vec[1:2] / (2-1))
            h = 1 - np.sqrt(f1 / (g + 1e-8))
            f2 = g * h  # objective 2
        else:
            raise NotImplementedError

        return [f1, f2]