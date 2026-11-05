import numpy as np
import cla_utils

#question1_b_1
m,n=48,12
t=np.arange(0,1+1/m,1/(m-1))
b=np.cos(4*t)
A=np.vander(t,n+1,increasing=True)
ATA=np.transpose(A)@A
ATb=np.transpose(A)@b

c_1=cla_utils.householder_solve(ATA, ATb.reshape(-1, 1))
print(np.linalg.norm(A @ c_1 - b.reshape(-1, 1)))

#question1_b_2
A_b=np.hstack((A,b.reshape(-1,1)))
R=cla_utils.GS_classical(A_b)
R1=R[:n+1,:n+1]
b_hat=R[:n+1,-1]
c_2=cla_utils.solve_U(R1,b_hat.reshape(-1,1))
print(np.linalg.norm(A @ c_2 - b.reshape(-1, 1)))

#question1_b_3
A_b1=np.hstack((A,b.reshape(-1,1)))
R=cla_utils.GS_modified(A_b1)
R1=R[:n+1,:n+1]
b_hat_1=R[:n+1,-1]
c_3=cla_utils.solve_U(R1,b_hat_1.reshape(-1,1))
print(b.shape)
print((A @ c_3).shape)
print(np.linalg.norm(A @ c_3 - b.reshape(-1, 1)))

#question1_b_4
_,n=A.shape
Ahat= np.hstack((A,b.reshape(-1,1)))
cla_utils.householder(Ahat)
Rhat=Ahat[:n,:n]
bhat=Ahat[:n,n]
c_4=cla_utils.solve_U(Rhat,bhat)
print(b.shape)
print((A @ c_4).shape)
print(np.linalg.norm(A @ c_4 - b))

print(c_1)
print(c_2)
print(c_3)
print(c_4)

def question2_solve(A,b,lam):
    m,n=A.shape
    I=np.identity(n)
    zero=np.zeros(n)
    Ahat=np.vstack((A,lam**(1/2)*I))
    bhat=np.concatenate((b,zero))
    x=cla_utils.householder_ls(Ahat,bhat)
    return x
from numpy import random
m, n, lam = 10, 20, 0.1

A = random.randn(m, n)
b = random.randn(m)
x = question2_solve(A, b, lam)

LHS = np.dot((np.transpose(A)@ A)+lam*np.identity(n),x)
RHS =np.dot(np.transpose(A),b)
print(np.linalg.norm(LHS - RHS))

m, n, lam = 10, 15, 10

A = random.randn(m, n)
b = random.randn(m)
x = question2_solve(A, b, lam)

LHS = np.dot((np.transpose(A)@ A)+lam*np.identity(n),x)
RHS =np.dot(np.transpose(A),b)
print(np.linalg.norm(LHS - RHS))

m, n, lam = 10, 44, 30

A = random.randn(m, n)
b = random.randn(m)
x = question2_solve(A, b, lam)

LHS = np.dot((np.transpose(A)@ A)+lam*np.identity(n),x)
RHS =np.dot(np.transpose(A),b)
print(np.linalg.norm(LHS - RHS))