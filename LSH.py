from scipy import integrate
import numpy as np
from typing import Union, Dict


class pstable:
    def __init__(self, r:int, dim:int, num_client:int, metric_dim:int=2, seed:int=1, num_perm:int=1024, 
                hashvalues: Dict[str, Union[np.ndarray, list]]=None):
        self.r = r
        self.seed = seed
        self.num_client = num_client
        self.metric_dim = metric_dim
        self.num_perm = num_perm
        self.dim = dim
        self.E2hashvalues = None
        self.R2hashvalues = None
        if hashvalues is not None:
            self.E2hashvalues = hashvalues.get("E2")
            self.R2hashvalues = hashvalues.get("R2")
        self.__init_permutations()
    
    def __len__(self):
        return len(self.hashvalues)

    def __init_permutations(self):
        self.gen = np.random.RandomState(self.seed)
        if self.metric_dim == 1:
            self.param_gen = self.__param_cauchy_gen
            self.f = self.f_cauchy
        elif self.metric_dim == 2:
            self.param_gen = self.__param_normal_gen
            self.f = self.f_gaussian
        else:
            raise ValueError("It can be only 1 or 2 for metric_dim!")
        self.a_1b = [self.param_gen(self.dim, 'E2') for _ in range(self.num_perm)]
        self.a_2b = [self.param_gen(self.dim, 'RH') for _ in range(self.num_perm)]

    def __param_normal_gen(self, dim, type) -> np.ndarray:
        mu, sigma = 0, 1
        if type == 'E2':
            b = []
            for i in range(self.num_client):
                b_i = self.gen.uniform(0, self.r)
                b.append(b_i)
        elif type == 'RH':
            b = self.gen.uniform(0, self.r)
        a = self.gen.normal(mu, sigma, dim)
        return a, b

    def E2lsh(self, x:Union[np.ndarray, list]) -> np.ndarray:
        self.E2hashvalues = np.array([(np.dot(a, x)+b)/self.r for a, b in self.a_1b])

        return self.E2hashvalues
    
    def E2lsh_new(self, x:Union[np.ndarray, list], client_id:int) -> np.ndarray:

        e2lsh = []
        for i in range(self.num_perm):
            a = self.a_1b[i][0]
            b = self.a_1b[i][1][client_id]
            e2lsh.append((np.dot(a, x)+b)/self.r)
        self.E2hashvalues = np.array(e2lsh)

        return self.E2hashvalues

    def R2lsh(self, x:Union[np.ndarray, list]) -> np.ndarray:
        self.R2hashvalues = np.array([np.heaviside(np.dot(a, x), 0) for a, _ in self.a_2b])
        
        return self.R2hashvalues

    def f_gaussian(self, x):
        return np.e**(-x**2/2)/np.sqrt(2*np.pi)

    def __pstableProb(self, x, c):
        return self.f(x/c)*(1-x/self.r)/c

    def p(self, x:Union[np.ndarray, list], y:Union[np.ndarray, list]) -> float:
        if type(x) != np.ndarray or type(y) != np.ndarray:
            x = np.array(x)
            y = np.array(y)
        c = np.linalg.norm(x-y, ord=self.metric_dim)
        v, err = integrate.quad(lambda t: self.__pstableProb(t, c), 0, self.r)
        return 2*v  
    

if __name__ == "__main__":
    LSH = pstable(1, 3, 2, 1, 10)
    print(LSH.a_2b)
    x = [1, 2.16, 3]
    y = [1.2, 2, 2.98]
    LSH.E2lsh(x)
    LSH.R2lsh(x)
    LSH.E2lsh(y, [1,1,1,1,1,1,1,1,1,1])
    print("*************************************** E2 LSH 1 ***************************************")
    print(LSH.E2hashvalues)
    print("*************************************** E2 LSH 2 ***************************************")
    LSH.E2lsh(y)
    print(LSH.E2hashvalues)
    print("*************************************** R2 LSH ***************************************")
    print(LSH.R2hashvalues)
    
    print(LSH.p(x, y))