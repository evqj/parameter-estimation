from SignalDetection import SignalDetection
from Experiment import Experiment
from SimplifiedThreePL import SimplifiedThreePL

if __name__ == '__main__':
        """Sets up the test case with SignalDetection objects."""
        sdt1 = SignalDetection(40, 10, 20, 30)
        sdt2 = SignalDetection(60, 20, 10, 40)
        sdt3 = SignalDetection(50, 15, 25, 35)
        sdt4 = SignalDetection(70, 20, 30, 45)
        sdt5 = SignalDetection(80, 25, 35, 55)
        sdt6 = SignalDetection(90, 25, 35, 55)
        exp = Experiment()
        exp.add_condition(sdt1, label="Condition A")
        exp.add_condition(sdt2, label="Condition B")
        exp.add_condition(sdt3, label="Condition C")
        exp.add_condition(sdt4, label="Condition D")
        exp.add_condition(sdt5, label="Condition E")
        exp.add_condition(sdt6, label="Condition F")
        model = SimplifiedThreePL(exp)
        dic = model.summary()
        print(dic)
        try:
            est_a = model.get_discrimination()
            print(est_a)
        except ValueError as e:
            print("ValueError throws")
        model.fit()
        est_b = model.get_discrimination()
        print(est_b)
        est_c = model.get_base_rate()
        print(est_c)
        predicts = model.predict([0.0, 1.0])
        print(predicts)
        negative = model.negative_log_likelihood([0.0, 1.0])
        print(negative)
    


