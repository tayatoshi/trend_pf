
# coding: utf-8

import numpy as np
import pandas as pd
import statsmodels.api as sm

class ParticleFilter:
    def __init__(self,y,number_of_particles,upsilon2,omega2):
        self.y=y
        self.length_of_time=y.shape[0]
        self.number_of_particles=number_of_particles
        self.upsilon2=upsilon2
        self.omega2=omega2
        self.filtered_value = []
        self.Trans=np.array([[1,1],[0,1]])# local trend level model
        # self.Trans=np.array([[1,0],[0,0]])# local level model
        np.random.seed(123)
    def init_particle(self):
        # x(i)_0|0
        particles = []
        predicts = []
        init_level=np.random.normal(7,1,self.number_of_particles)
        init_trend=np.random.normal(0,0.1,self.number_of_particles)
        init=np.array([init_level,init_trend]).T
        particles.append(init)
        predicts.append(init)
        return({'particles':particles,'predicts':predicts})

    def get_likelihood(self,ensemble,t):
        #正規分布を仮定
        likelihoodes = np.zeros(self.number_of_particles)
        for n in range(self.number_of_particles):
            likelihoodes[n] = (1/np.sqrt(2*np.pi*self.omega2))*np.exp(-(self.y[t]-np.dot(np.array([1,1]).reshape(1,2),ensemble[t][n].reshape(2,1)))**2/2*self.omega2)
        return(likelihoodes)

    def one_predict(self,ensemble,t):
        Sigma = np.random.multivariate_normal([0,0],[[self.upsilon2[0],0],[0,self.upsilon2[1]]],self.number_of_particles)
        one_predicts = np.dot(self.Trans,ensemble[t].T)+Sigma.T
        return(one_predicts.T)

    def filtering(self,ensemble,t):
        # x(i)_t|t
        likelihood=self.get_likelihood(ensemble,t)
        beta=likelihood/np.sum(likelihood) #尤度の比率
        filtering_value=np.sum(beta.reshape(self.number_of_particles,1)*ensemble[t],axis=0)
        return({'beta':beta,'filtering_value':filtering_value})

    def resumpling(self,ensemble,weight):
        sample0=np.random.choice(ensemble[:,0],p=weight,size=self.number_of_particles)
        sample1=np.random.choice(ensemble[:,1],p=weight,size=self.number_of_particles)
        sample = np.array([sample0,sample1])
        return(sample.T)

    def simulate(self):
        particles=self.init_particle()['particles']
        predicts=self.init_particle()['predicts']
        filtered_value=[]
        filtered_value.append(np.sum(particles[0],axis=0)/self.number_of_particles)
        for t in np.arange(1,self.length_of_time):
            print("\r calculating... t={}".format(t), end="")
            #一期先予測
            predicts.append(self.one_predict(particles,t-1))
            #フィルタリング
            filtered=self.filtering(predicts,t)
            filtered_value.append(filtered['filtering_value'])
            resumple=self.resumpling(predicts[t],filtered['beta'])
            particles.append(resumple)
        return({'particles':np.array(particles),'predicts':np.array(predicts),'filtered_value':np.array(filtered_value)})

    def forecast(self,result,time=10):
        Forecasts_ens = []
        Forecasts_mean = []
        ensemble= result['particles'][-1]
        for t in range(time):
            if t == 0:
                Sigma = np.random.multivariate_normal([0,0],[[self.upsilon2[0],0],[0,self.upsilon2[1]]],self.number_of_particles)
                Forecasts_ens.append(np.dot(ensemble.reshape(self.number_of_particles,2),self.Trans.T)+Sigma)
                Forecasts_mean.append(np.mean(Forecasts_ens[t],axis = 0))
            else:
                Sigma = np.random.multivariate_normal([0,0],[[self.upsilon2[0],0],[0,self.upsilon2[1]]],self.number_of_particles)
                Forecasts_ens.append(np.dot(Forecasts_ens[t-1],self.Trans.T)+Sigma)
                Forecasts_mean.append(np.mean(Forecasts_ens[t],axis = 0))
        return({'Forecasts_ens':np.array(Forecasts_ens),'Forecasts_mean':np.array(Forecasts_mean)})

def rmse(predic,test):
    mse = np.sum(((predic-test)**2))/predic.shape[0]
    RMSE = np.sqrt(mse)
    return(RMSE)

print('ok!')
