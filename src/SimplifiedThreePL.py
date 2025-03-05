import numpy
import scipy
from Experiment import Experiment

class SimplifiedThreePL:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

        if len(self.experiment.conditions) == 0:
            raise ValueError("Experiment has no condition")
        self.default_difficulties_parameters = [2, 1, 0, -1, -2]
        self.theta = 0  # person parameter fixed at 0
        self._base_rate = None        # base rate parameter c
        self._logit_base_rate = None  # logit of the base rate parameter q
        self._discrimination = None   # discrimnation parameter a
        self._is_fitted = False
    

    def summary(self):
        # Implement a summary method that returns a dictionary with the following keys:
        #    n_total: total number of trials
        #    n_correct: number of correct trials
        #    n_incorrect: number of incorrect trials
        #    n_conditions: number of conditions
        n_total = 0
        n_correct = 0
        n_incorrect = 0
        for condition in self.experiment.conditions:
            n_total += condition.n_total_responses()
            n_correct += condition.n_correct_responses()
            n_incorrect += condition.n_incorrect_responses()
        return {
            'n_total': n_total,
            'n_correct': n_correct,
            'n_incorrect': n_incorrect,
            'n_conditions': len(self.experiment.conditions)
        }


    def predict(self, parameters):
        # Implement a predict method that takes parameters as input and returns the probability of a correct response in each condition, given the parameters.
        a, q = parameters
        c = self.__calculate_base_rate_parameter(q)
        predictions = []
        condition_count =  len(self.experiment.conditions)
        if condition_count > len(self.default_difficulties_parameters):
            condition_count = len(self.default_difficulties_parameters)  # we only check up to number of difficulties parameters
        current = 0
        while current < condition_count:
            bi = self.default_difficulties_parameters[current]
            p_correct = self.__calculate_probablity_of_a_correct_response(a, bi, c)
            predictions.append(p_correct)
            current += 1
        return predictions


    def negative_log_likelihood(self, parameters) -> float:
        # Implement a negative_log_likelihood method that computes the negative log-likelihood of the data given the parameters.
        a, q = parameters
        c = self.__calculate_base_rate_parameter(q)
        neg_log_likelihood = 0.0
        condition_count =  len(self.experiment.conditions)
        if condition_count > len(self.default_difficulties_parameters):
            condition_count = len(self.default_difficulties_parameters)  # we only check up to number of difficulties parameters
        current = 0
        while current < condition_count:
            bi = self.default_difficulties_parameters[current]
            p_correct = self.__calculate_probablity_of_a_correct_response(a, bi, c)
            n_correct = self.experiment.conditions[current].n_correct_responses()
            n_incorrect = self.experiment.conditions[current].n_incorrect_responses()
            neg_log_likelihood -= (n_correct * numpy.log(p_correct) + n_incorrect * numpy.log(1 - p_correct))
            current += 1
        return neg_log_likelihood


    def fit(self):
        # Implements maximum likelihood estimation to find the best-fitting discrimination parameter
        # base rate parameter.
        initial_guess = [0.0, 0.0]   # initial guess for a and q
        result = scipy.optimize.minimize(self.negative_log_likelihood, initial_guess)
        self._discrimination, self._logit_base_rate = result.x
        self._base_rate = 1 / (1 + numpy.exp(-self._logit_base_rate))
        self._is_fitted = True

    
    def get_discrimination(self):
        # Returns estimate of discrimination parameter alpha.
        if self._is_fitted:
            return self._discrimination
        raise ValueError("Model is not fitted yet.")


    def __calculate_probablity_of_a_correct_response(self, a, bi, c) -> float:
        p_correct = c + (1 - c) / (1 + numpy.exp(-a * (self.theta - bi)))
        return p_correct


    def __calculate_base_rate_parameter(self, q) -> float:
        c = 1 / (1 + numpy.exp(-q))
        return c


    def get_base_rate(self):
        # Implement a get_base_rate method that returns the estimate of the base rate parameter c (not q)
        if self._is_fitted:
            return self._base_rate
        raise ValueError("Model is not fitted yet.")
       
    
    def get_logit_base_rate(self):
        if self._is_fitted:
            return self._logit_base_rate
        raise ValueError("Model is not fitted yet.")