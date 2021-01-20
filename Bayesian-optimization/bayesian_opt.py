import random
import math
import numpy as np
import pandas as pd
import scipy.stats as scs
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C





"""
    The script for BayesOptimization
"""


class BO:
    
    def __init__(self, search_space, list_of_parameters_names, maximize):

        self.dict_of_means = {}
        self.list_of_parameters_names = list_of_parameters_names
        self.search_space = search_space
        for key in list_of_parameters_names:

            self.dict_of_means[key] = [float(search_space[key][0][1]+search_space[key][0][0])/2.0,
                                       float(search_space[key][0][1]-search_space[key][0][0])/2.0]
    
    
        # Instantiate a Gaussian Process model
        self.kernel =  RBF(5, (1e-2, 1e2))*C(1, (1e-2, 1e2))+ WhiteKernel(noise_level=0.2)
    
        self.maximize = maximize
        

        self.generate_meshes()
        
        self.parameters_and_loss_dict = {}
        for item in self.list_of_parameters_names:
            self.parameters_and_loss_dict[item] = []
        self.parameters_and_loss_dict['loss'] = []
        
    def generate_meshes(self):
        
        # Create the list of ranges for the search space (start, end, number_of_points)
        list_of_ranges = []
        list_of_shapes = []
        list_of_ranges_true =[]

        search_space_normalized = {}
        for key in self.list_of_parameters_names:

            search_space_normalized[key] = [(self.search_space[key][0][0]-self.dict_of_means[key][0])/self.dict_of_means[key][1],
                                       (self.search_space[key][0][1]-self.dict_of_means[key][0])/self.dict_of_means[key][1],
                                       (self.search_space[key][0][2])]
            #print(search_space_normalized)
            list_of_ranges_true.append (np.linspace(*self.search_space[key][0]))
            list_of_ranges.append(np.linspace(*search_space_normalized[key]))
            list_of_shapes.append(self.search_space[key][0][2])

        # Create a meshgrid from the list of ranges for the searchspace
        meshgrid_linspace = np.meshgrid(*(list_of_ranges),indexing='ij')

        reshape_param = np.product(np.shape(meshgrid_linspace[0]))
        meshgrid_linspacer = []
        
        for mlinsp in meshgrid_linspace:
            meshgrid_linspacer.append(np.reshape(mlinsp,reshape_param))

        # meshgrid for GP prediction
        self.meshgrid_linspacer_stack = np.stack(meshgrid_linspacer,axis=1)
        self.list_of_shapes = list_of_shapes
        self.list_of_ranges_true = list_of_ranges_true
        self.list_of_ranges = list_of_ranges
        return 
    
    
    def normalize_current_values(self):
        normalized_param= []
        # Create a list of estimation parameters, iterate over all, skip the loss from the list
        list_of_parameters = []
        for key in self.list_of_parameters_names:
            normalized_param=[(float(param)-self.dict_of_means[key][0])/self.dict_of_means[key][1] for param in self.parameters_and_loss_dict[key]]

            list_of_parameters.append(np.atleast_2d(normalized_param).T)

        list_of_parameters_stack = np.stack(list_of_parameters,axis=1)
        

        return list_of_parameters_stack, list_of_parameters
        
        
        
    def bayes_opt(self):

        if len(self.parameters_and_loss_dict[self.list_of_parameters_names[0]])>0:
            
            # Generate function prediction from the parameters and the loss (run bayesian regression)
            loss_predicted,sigma,loss_evaluated = self.generate_prediction()

            # Calculate expected improvement (finding the maximum of the information gain function)
            expected_improvement = self.calculate_expected_improvement(loss_predicted, sigma, loss_evaluated)

            # Find the parameter values for the maximum values of the information gain function
            next_parameter_values = self.find_next_parameter_values(expected_improvement)

        else:

            next_parameter_values = {}
            for name in self.list_of_parameters_names:
                next_parameter_values[name] = self.search_space[name][1](self.search_space[name][0][0]+(self.search_space[name][0][1]-self.search_space[name][0][0])*np.random.uniform(0,1))

            loss_predicted = None
            sigma=None
            loss_evaluated=None
            expected_improvement=None   


        parameters_and_loss_df=pd.DataFrame.from_dict(self.parameters_and_loss_dict,orient='index').transpose()   
        
        # Write the results into the dataframe
        list_of_next_values = []

        for item in self.list_of_parameters_names:
            # Add all parameters to the list, so that they can be appended to the dataframe
            # While iterating convert to the preffered datatype
            list_of_next_values.append(self.search_space[item][1](round(next_parameter_values[item],5)))
            next_parameter_values[item] = (self.search_space[item][1](round(next_parameter_values[item],5)))

        # Check that the output is not repeated (can happen as we are using white noize kernel) in this case generate point at random
        while (parameters_and_loss_df[self.list_of_parameters_names]== list_of_next_values).all(1).any():
            list_of_next_values = []
            for item in self.list_of_parameters_names:
                rand_value = self.search_space[item][1](round(random.uniform(self.search_space[item][0][0], self.search_space[item][0][1]),5))
                next_parameter_values[item] = rand_value
                list_of_next_values.append(rand_value)


        for item in self.list_of_parameters_names:
            self.parameters_and_loss_dict[item].append(next_parameter_values[item])
        
        
        return next_parameter_values, loss_predicted, sigma, expected_improvement


    def update_loss(self, value):
        
        self.parameters_and_loss_dict['loss'].append(value)
        
        return
    

    def fit_gp(self, list_of_parameters_stack, list_of_parameters,loss_evaluated):

        gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=200)

        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp.fit(np.reshape(list_of_parameters_stack,(-1,len(list_of_parameters))), loss_evaluated)
    
        # Make the prediction on the meshed x-axis (ask for MSE as well)
        loss_predicted, sigma = gp.predict(self.meshgrid_linspacer_stack, return_std=True)

        loss_predicted = np.reshape(loss_predicted,self.list_of_shapes)
        sigma = np.reshape(sigma,self.list_of_shapes)

        return loss_predicted, sigma 


    def generate_prediction(self):

        list_of_parameters_stack, list_of_parameters = self.normalize_current_values()
        loss_evaluated = self.parameters_and_loss_dict['loss']
        if len(self.parameters_and_loss_dict['loss'])>1:
            loss_evaluated = (self.parameters_and_loss_dict['loss']-np.mean(self.parameters_and_loss_dict['loss']))/(np.std(self.parameters_and_loss_dict['loss'])+1e-6)

        loss_predicted, sigma  = self.fit_gp(list_of_parameters_stack, list_of_parameters, loss_evaluated)

        return loss_predicted,sigma,loss_evaluated


    def calculate_expected_improvement(self, loss_predicted, sigma, loss_evaluated):

        # Calculate the expected improvement
        eps = 1e-6
        num =(loss_predicted-max(loss_evaluated)-eps)
        Z=num/sigma
        expected_improvement = num*scs.norm(0,1).cdf(Z)+sigma*scs.norm(0,1).pdf(Z)
        expected_improvement[sigma==0.0] = 0.0

        return expected_improvement


    def find_next_parameter_values(self, expected_improvement):
        
        
        if self.maximize:
            index = np.where(expected_improvement==np.amax(expected_improvement))
        else:
            index = np.where(expected_improvement==np.amin(expected_improvement))

        next_parameter_values = {}
        # Iterate over all parameter values and find those corresponding to maximum EI
        for idx, parameter in enumerate(self.list_of_parameters_names):

            # Since more than one value can be have max at EI,select one at random
            x = int(np.random.uniform(0,len(index[idx])))
            next_parameter_values[parameter] = self.list_of_ranges_true[idx][index[idx][x]]

        return next_parameter_values



