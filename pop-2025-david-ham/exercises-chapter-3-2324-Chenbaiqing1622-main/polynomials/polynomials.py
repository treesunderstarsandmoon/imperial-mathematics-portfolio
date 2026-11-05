from numbers import Number
from numbers import Integral


class Polynomial:

    def __init__(self, coefs):
        self.coefficients = coefs

    def degree(self):
        return len(self.coefficients) - 1

    def __str__(self):
        coefs = self.coefficients
        terms = []

        if coefs[0]:
            terms.append(str(coefs[0]))
        if self.degree() and coefs[1]:
            terms.append(f"{'' if coefs[1] == 1 else coefs[1]}x")

        terms += [f"{'' if c == 1 else c}x^{d}"
                  for d, c in enumerate(coefs[2:], start=2) if c]

        return " + ".join(reversed(terms)) or "0"

    def __repr__(self):
        return self.__class__.__name__ + "(" + repr(self.coefficients) + ")"

    def __eq__(self, other):

        return isinstance(other, Polynomial) and\
             self.coefficients == other.coefficients

    def __add__(self, other):

        if isinstance(other, Polynomial):
            common = min(self.degree(), other.degree()) + 1
            coefs = tuple(a + b for a, b in zip(self.coefficients,
                                               other.coefficients))
            coefs += self.coefficients[common:] + other.coefficients[common:]

            return Polynomial(coefs)

        elif isinstance(other, Number):
            return Polynomial((self.coefficients[0] + other,)
                              + self.coefficients[1:])

        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other
    
    def __sub__(self,other):
        if isinstance(other, Polynomial):
            common = max(self.degree(), other.degree())

            while self.degree()<common:
                self.coefficients+=(0)
            while other.degree()<common:
                other.coefficients+=(0)


            coefs = tuple(a - b for a, b in zip(self.coefficients,
                                               other.coefficients))
            
            return Polynomial(coefs)

        elif isinstance(other, Number):
            return Polynomial((self.coefficients[0] - other,)
                              + self.coefficients[1:])

        else:
            return NotImplemented
        
    def __rsub__(self, other):
        return -1* (self - other)
    
    def __mul__(self, other):
        if isinstance(other, Polynomial):
            m = self.degree() + other.degree()
            coefs = [0] * (m + 1)
            for i in range(len(self.coefficients)):
                for j in range(len(other.coefficients)):
                    coefs[i + j] += self.coefficients[i] * other.coefficients[j]

            while coefs and coefs[-1] == 0:
                coefs.pop()
            return Polynomial(tuple(coefs))  
        elif isinstance(other, Number):
            coefs = [a * other for a in self.coefficients]
            return Polynomial(tuple(coefs)) 
        else:
            return NotImplemented
    def __rmul__(self, other):
        if isinstance(other, Polynomial):
            return self * other
        elif isinstance(other, Number):
            coefs = tuple(a * other for a in self.coefficients)
            return Polynomial(coefs)
        else:
            return NotImplemented
        
    def __pow__(self,n):
        if isinstance(n, Integral):
            new = self
            for i in range(n-1):
                new = new*self
            return new
        else:
            return NotImplemented



    def __call__(self,other):
        if isinstance(other,Number):
            answer=0
            for i in range(self.degree()+1):
                answer+= self.coefficients[i]*other**i 
            return answer  
        else:
            return NotImplemented
        

    def dx(self):
        if self.degree() == 0 :
            return Polynomial((0,))
        else:
            self.coef=[]
            for i in range(1,self.degree()+1):
                self.coef.append(self.coefficients[i]*i)
            a=tuple(self.coef)
            return Polynomial(a)
    
    
def derivative(poly) :
    return poly.dx() 


