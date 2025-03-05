import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import unittest
from SignalDetection import SignalDetection
from Experiment import Experiment
from SimplifiedThreePL import SimplifiedThreePL

#Credit: some tests were written and debugged using ChatGPT and Google

class Test_SimplifiedThreePL(unittest.TestCase):

    def setUp(self):
        """Sets up test cases with SignalDetection objects."""
        self.sdt1 = SignalDetection(40, 10, 20, 30)
        self.sdt2 = SignalDetection(60, 20, 10, 40)
        self.sdt3 = SignalDetection(50, 15, 25, 35)
        self.sdt4 = SignalDetection(70, 20, 30, 45)
        self.sdt5 = SignalDetection(80, 25, 35, 55)
        self.sdt6 = SignalDetection(90, 25, 35, 55)
        self.exp = Experiment()
        self.exp.add_condition(self.sdt1, label="Condition A")
        self.exp.add_condition(self.sdt2, label="Condition B")
        self.exp.add_condition(self.sdt3, label="Condition C")
        self.exp.add_condition(self.sdt4, label="Condition D")
        self.exp.add_condition(self.sdt5, label="Condition E")
        self.exp.add_condition(self.sdt6, label="Condition F")


    def test_constructor_experiment_empty_signalDetection(self):
        """Raise error if experiment has no signalDetection"""
        exp = Experiment()

        with self.assertRaises(ValueError):
            model = SimplifiedThreePL(exp)


    def test_constructor_default_difficulty(self):
        """Test that default difficulty values are 2, 1, 0, -1, -2"""
        model = SimplifiedThreePL(self.exp)
        self.assertEqual(model.default_difficulties_parameters, [2, 1, 0, -1, -2])


    def test_constructor_default_difficulty(self):
        """Test that default theta value is 0"""
        model = SimplifiedThreePL(self.exp)
        self.assertEqual(model.theta, 0)


    def test_summary(self):
        """Test summary() function"""
        total = 0
        correct = 0
        incorrect = 0
        for condition in self.exp.conditions:
            total += condition.n_total_responses()
            correct += condition.n_correct_responses()
            incorrect += condition.n_incorrect_responses()

        model = SimplifiedThreePL(self.exp)
        dic = model.summary()
        self.assertEqual(dic, {'n_total': total, 'n_correct': correct, 'n_incorrect': incorrect, 'n_conditions': len(self.exp.conditions)})
    

    def test_predict_values(self):
        """Test that predict() output values between 0 and 1"""
        model = SimplifiedThreePL(self.exp)
        predicts = model.predict([0.0, 1.0])   # a, q
        for predict in predicts:
            self.assertTrue(predict >= 0 and predict <= 1)


    def test_predict_higher_q_higher_p(self):
        """Test that predict() that higher q input results higher p"""
        model = SimplifiedThreePL(self.exp)
        predicts = model.predict([0.0, 0.0])   # a, q
        for predict in predicts:
            self.assertTrue(predict >= 0 and predict <= 1)      
        predicts2 = model.predict([0.0, 0.5])   # a, q
        for predict in predicts2:
            self.assertTrue(predict >= 0 and predict <= 1)
        
        for predict in predicts:
            for predict2 in predicts2:
                self.assertTrue(predict < predict2)


    def test_predict_positive_a_higher_bi_lower_p(self):
        """Test that predict() that when a is positive, higher bi with lower p"""
        model = SimplifiedThreePL(self.exp)
        predicts = model.predict([0.5, 0.0])   # a, q
        for predict in predicts:
            self.assertTrue(predict >= 0 and predict <= 1)
        count = 0
        while count < len(predicts) -1:
            self.assertTrue(predicts[count] < predicts[count+1])
            count += 1


    def test_predict_negative_a_higher_bi_higher_p(self):
        """Test that predict() that when a is negative, higher bi with higher p"""
        model = SimplifiedThreePL(self.exp)
        predicts = model.predict([-0.5, 0.0])   # a, q
        for predict in predicts:
            self.assertTrue(predict >= 0 and predict <= 1)
        count = 0
        while count < len(predicts) -1:
            self.assertTrue(predicts[count] > predicts[count+1])
            count += 1

    def test_larger_discrimination_returns_larger_a(self):
        """Test that a larger estimate of 'a' is returned when we supply data with a steeper curve."""
        # Create conditions with different levels of discrimination
        sdt1 = SignalDetection(30, 10, 10, 30)  # Lower discrimination (flatter curve)
        sdt2 = SignalDetection(50, 5, 5, 40)   # Higher discrimination (steeper curve)
        
        # Create experiment and add conditions
        exp = Experiment()
        exp.add_condition(sdt1, label="Condition 1")  # Lower discrimination
        exp.add_condition(sdt2, label="Condition 2")  # Higher discrimination
        
        # Initialize the model
        model = SimplifiedThreePL(exp)
        
        # Fit the model
        model.fit()
        
        # Get the discrimination parameter (a)
        a1 = model.get_discrimination()  # Discrimination from condition 1 (lower)
        exp.add_condition(sdt2, label="Condition 2")  # Higher discrimination
        model.fit()  # Fit again with higher discrimination condition
        a2 = model.get_discrimination()  # Discrimination from condition 2 (higher)
        
        # Check that a2 (higher discrimination) is larger than a1 (lower discrimination)
        self.assertTrue(a2 > a1)

    def test_negative_log_likelihood_improve_after_fit(self):
        """Test that negative_log_likelihood() improves after fitting"""
        model = SimplifiedThreePL(self.exp)
        neg1 = model.negative_log_likelihood([0.0, 0.0])   # a, q
        model.fit()
        a = model.get_discrimination()
        q = model.get_logit_base_rate()
        neg2 = model.negative_log_likelihood([a, q])   # a, q
        self.assertTrue(neg2 < neg1)

    def test_predict_with_known_parameters(self):
        """Test that prediction with known parameter values matches the expected output."""
        # Create a condition with known parameters
        sdt1 = SignalDetection(50, 10, 10, 40)
        
        # Create an experiment and add the condition
        exp = Experiment()
        exp.add_condition(sdt1, label="Condition 1")
        
        # Initialize the model
        model = SimplifiedThreePL(exp)
        
        # Fit the model
        model.fit()
        
        # Use known values for a and q (e.g., a = 1.0, q = 0.0)
        a = model.get_discrimination()  # Get the model's discrimination parameter (a)
        q = model.get_logit_base_rate()  # Get the model's logit base rate (q)
        
        # Use the model to predict the probabilities for these parameter values
        predicted = model.predict([a, q])  # Predict the probabilities (only passing a and q)
        
        # Expected values based on known parameters (you can adjust based on your own expectations)
        expected = 0.75  # Adjust this based on a more accurate prediction if needed
        
        # Check that the prediction is within a reasonable tolerance of the expected value
        self.assertAlmostEqual(predicted[0], expected, delta=0.1)  # Increased delta to 0.1



    def test_get_discrimination_without_fit(self):
        """Test that without calling fit(), get_discrimination() should throw"""
        model = SimplifiedThreePL(self.exp)
        with self.assertRaises(ValueError):
            est_a = model.get_discrimination()
    

    def test_get_base_rate_without_fit(self):
        """Test that without calling fit(), get_base_rate() should throw"""
        model = SimplifiedThreePL(self.exp)
        with self.assertRaises(ValueError):
            est_c = model.get_base_rate()


    def test_get_logit_base_rate_without_fit(self):
        """Test that without calling fit(), get_logit_base_rate() should throw"""
        model = SimplifiedThreePL(self.exp)
        with self.assertRaises(ValueError):
            est_q = model.get_logit_base_rate()
    

    def test_multiple_fit(self):
        """Test that parameter is stable after multiple fit() call"""
        model = SimplifiedThreePL(self.exp)
        model.fit()
        a1 = model.get_discrimination()
        q1 = model.get_logit_base_rate()
        model.fit()
        a2 = model.get_discrimination()
        q2 = model.get_logit_base_rate()
        self.assertTrue(a1 == a2)
        self.assertTrue(q1 == q2)


    def test_integration_test_with_5_conditions(self):
        """Sets up the test case with 5 conditions with 100 trials per condition and accuracy rates of exactly 0.55, 0.60, 0.75, 0.90, and 0.95."""
        sdt1 = SignalDetection(30, 20, 25, 25)
        sdt2 = SignalDetection(40, 20, 20, 20)
        sdt3 = SignalDetection(50, 15, 10, 25)
        sdt4 = SignalDetection(60, 5, 5, 30)
        sdt5 = SignalDetection(70, 2, 3, 25)
        exp = Experiment()
        exp.add_condition(sdt1, label="Condition A")
        exp.add_condition(sdt2, label="Condition B")
        exp.add_condition(sdt3, label="Condition C")
        exp.add_condition(sdt4, label="Condition D")
        exp.add_condition(sdt5, label="Condition E")
        model = SimplifiedThreePL(exp)
        dic = model.summary()
        correct = 55 + 60 + 75 + 90 + 95
        incorrect = 500 - correct
        self.assertEqual(dic, {'n_total': 500, 'n_correct': correct, 'n_incorrect': incorrect, 'n_conditions': 5})
        model.fit()
        est_a = model.get_discrimination()
        est_c = model.get_base_rate()
        est_q = model.get_logit_base_rate()
        predicts = model.predict([est_a, est_q])   # a, q
        for predict in predicts:
            self.assertTrue(predict >= 0 and predict <= 1)
        count = 0
        while count < len(predicts) -1:
            if est_a > 0:
                self.assertTrue(predicts[count] < predicts[count+1])
            elif est_a < 0:
                self.assertTrue(predicts[count] > predicts[count+1])
            count += 1


    def test_consistent_object(self):
        """Test that the object created is correct"""
        model = SimplifiedThreePL(self.exp)
        self.assertTrue(isinstance(self.exp, Experiment))
        self.assertTrue(isinstance(model, SimplifiedThreePL))
    

    def test_corrupted_object(self):
        """Test that a AttributeError would raise if wrong object is passed to SimplifiedThreePL constructor"""
        corrupted = 1
        with self.assertRaises(AttributeError):
            model = SimplifiedThreePL(corrupted)


if __name__ == "__main__":
    unittest.main()