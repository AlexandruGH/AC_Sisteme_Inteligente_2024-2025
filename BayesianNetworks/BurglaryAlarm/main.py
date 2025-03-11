from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


if __name__ == "__main__":
    alarm_model = BayesianNetwork(
        [
            ('Burglary', 'Alarm'),
            ('Earthquake', 'Alarm'),
            ('Alarm', 'JohnCall'),
            ('Alarm', 'MaryCall')
        ])

    cpd_burglary = TabularCPD(
        variable='Burglary',
        variable_card=2,
        values=[[0.999], [0.001]]
    )

    cpd_earthquake = TabularCPD(
        variable='Earthquake',
        variable_card=2,
        values=[[0.998], [0.002]]
    )

    cpd_alarm = TabularCPD(
        variable='Alarm',
        variable_card=2,
        values=[
            [0.999, 0.71, 0.06, 0.05],
            [0.001, 0.29, 0.94, 0.95]
        ],
        evidence=['Burglary', 'Earthquake'],
        evidence_card=[2, 2],
    )

    cpd_john_call = TabularCPD(
        variable='JohnCall',
        variable_card=2,
        values=[
            [0.95, 0.1],
            [0.05, 0.9]
        ],
        evidence=['Alarm'],
        evidence_card=[2],
    )

    cpd_mary_call = TabularCPD(
        variable='MaryCall',
        variable_card=2,
        values=[
            [0.99, 0.3],
            [0.01, 0.7]
        ],
        evidence=['Alarm'],
        evidence_card=[2]
    )

    alarm_model.add_cpds(
        cpd_burglary, cpd_earthquake, cpd_alarm, cpd_john_call, cpd_mary_call
    )

    print(alarm_model.nodes())
    print(alarm_model.edges())

    alarm_infer = VariableElimination(alarm_model)

    print(alarm_infer.query(variables=['JohnCall', 'MaryCall', 'Alarm', 'Burglary', 'Earthquake'], evidence={'Burglary': 1}, show_progress=True)) # E explains away the Alarm

    print(alarm_model.local_independencies('JohnCall'))

    print(alarm_model.get_independencies())