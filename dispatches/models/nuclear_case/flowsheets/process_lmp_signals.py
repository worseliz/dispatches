import pandas as pd
import numpy as np
from pyomo.environ import Param, RangeSet
import json


def append_lmp_signal(m,
                      signal_source="ARPA_E",
                      signal_name="MiNg_$100_CAISO"):
    if signal_source == "ARPA_E":
        raw_data = pd.read_excel(
            "FLECCS_Price_Series_Data_01_20_2021.xlsx",
            sheet_name="2035 - NREL")

        price_all = raw_data[signal_name].tolist()

    elif signal_source == "RTS_GMLC":
        raw_data = np.load("rts_results_all_prices.npy")
        price_all = raw_data.tolist()

    lmp_data_full_year = np.array([price_all[24 * i: 24 * (i + 1)]
                                   for i in range(364)])

    # m.set_days = RangeSet(lmp_data_full_year.shape[0])
    # num_days = 364
    # m.set_days = RangeSet(num_days)
    # m.set_hours = RangeSet(24)
    #
    # m.weights_days = Param(m.set_days, initialize={
    #     d: (364 / num_days) for d in m.set_days})
    #
    # m.LMP = Param(m.set_hours, m.set_days,
    #               initialize={
    #                   (t, d): lmp_data_full_year[d - 1, t - 1]
    #                   for t in m.set_hours for d in m.set_days})

    with open('lmp_price_signal.json') as fp:
        full_lmp_data = json.load(fp)

    m.set_days = RangeSet(20)
    m.set_hours = RangeSet(24)

    m.weights_days = Param(m.set_days, initialize={
        d: full_lmp_data["0"]["2021"][str(d)]["num_days"] for d in m.set_days})

    m.LMP = Param(m.set_hours, m.set_days,
                  initialize={
                      (t, d): full_lmp_data["0"]["2021"][str(d)][str(t)]
                      for t in m.set_hours for d in m.set_days})


def append_raven_lmp_signal(m, scenarios, years,
                            plant_life=30,
                            discount_rate=0.08,
                            tax_rate=0.2,
                            prob_vec=None):
    if plant_life < len(years):
        raise Exception("Plant lifetime is lower than |set_years|!")

    with open('lmp_price_signal.json') as fp:
        full_lmp_data = json.load(fp)

    # Append the set of hours, days, years, and scenarios
    print("=" * 80)
    print("        NOTE: NUMBER OF CLUSTERS/REP. DAYS IS ASSUMED TO BE 20!")

    m.LMP = {s: {y: {c: {h: full_lmp_data[str(s)][str(y)][str(c)][str(h)]
                         for h in range(1, 25)}
                     for c in range(1, 21)}
                 for y in years}
             for s in scenarios}

    m.set_hours = RangeSet(24)
    m.set_days = RangeSet(20)
    m.set_years = years
    m.set_scenarios = scenarios

    # Load the parameters to the model
    m.tax_rate = tax_rate
    m.plant_life = plant_life
    m.discount_rate = discount_rate

    # Number of days represented by each cluster
    m.weights_days = {y: {c: full_lmp_data[str(0)][str(y)][str(c)]["num_days"]
                          for c in range(1, 21)}
                      for y in years}

    # Compute the coefficients for each year
    if len(years) != plant_life:
        print("        NOTE: # Years != Plant lifetime. "
              "Using the previous year's LMP for missing years!")

    years_vec = [y - years[0] + 1 for y in years]
    years_vec.append(plant_life + 1)
    m.weights_years = {y: sum(1 / (1 + discount_rate) ** i
                              for i in range(years_vec[j], years_vec[j + 1]))
                       for j, y in enumerate(years)}

    # Weights for each scenario. If the probability vector is not
    # specified, we assume each scenario is equally likely.
    if prob_vec is None:
        print("        NOTE: ALL SCENARIOS ARE ASSUMED TO BE EQUALLY LIKELY")
        m.weights_scenarios = {s: 1 / len(scenarios) for s in scenarios}
    elif len(prob_vec) != len(scenarios):
        raise Exception("Probability vector is not completely specified!")
    elif sum(prob_vec) != 1:
        raise Exception("Probability vector does not add up to 1!")
    else:
        m.weights_scenarios = {s: prob_vec[j] for j, s in enumerate(scenarios)}

    print("=" * 80)
