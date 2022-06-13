# General python imports
import matplotlib.pyplot as plt
import time

# Pyomo imports
from pyomo.environ import (Constraint,
                           Var,
                           ConcreteModel,
                           Expression,
                           Objective,
                           NonNegativeReals,
                           Block,
                           Param,
                           maximize,
                           SolverFactory)

# IDAES imports
from idaes.core import FlowsheetBlock

# Import additional functions
from process_lmp_signals import append_lmp_signal, append_raven_lmp_signal


def build_ne_flowsheet(m):
    m.fs = FlowsheetBlock(default={"dynamic": False})

    # Declare variables -- These will be needed later for the cashflows
    m.fs.np_power = Var(within=NonNegativeReals,
                        doc="Power produced by the nuclear plant (MW)")
    m.fs.np_to_grid = Var(within=NonNegativeReals,
                          doc="Power from NP to the grid (MW)")
    m.fs.np_to_electrolyzer = Var(within=NonNegativeReals,
                                  doc="Power from NP to electrolyzer (MW)")
    m.fs.h2_production = Var(within=NonNegativeReals,
                             doc="Hydrogen production rate (kg/hr)")
    m.fs.tank_holdup = Var(within=NonNegativeReals,
                           doc="Hydrogen holdup in the tank (kg)")
    m.fs.tank_holdup_previous = Var(within=NonNegativeReals,
                                    doc="Hold at the beginning of the period (kg)")
    #let's get rid of this and focus on the turbine
    #m.fs.h2_to_pipeline = Var(within=NonNegativeReals,
    #                          doc="Hydrogen flowrate to the pipeline (kg/hr)")
    m.fs.h2_to_turbine = Var(within=NonNegativeReals,
                             doc="Hydrogen flowrate to the turbine (kg/hr)")
    m.fs.h2_turbine_power = Var(within=NonNegativeReals,
                                doc="Power production from H2 turbine (MW)")
    m.fs.net_power = Var(within=NonNegativeReals,
                         doc="Net power to the grid (MW)")
    m.fs.pem_capacity = Var(within=NonNegativeReals,
                            doc="Maximum capacity of the PEM electrolyzer (in MW)")

    # Fix power production from NE plant to 1 GW
    # This way we can eliminate the NPP costs
    m.fs.np_power.fix(1000)
   

    # Declare Constraints
    m.fs.np_power_balance = Constraint(
        expr=m.fs.np_power == m.fs.np_to_grid + m.fs.np_to_electrolyzer,
        doc="Power balance at the nuclear power plant"
    )
    # Compute the hydrogen production rate
    # H-tec design: 54.517 kW-hr/kg of hydrogen
    m.fs.calc_h2_production_rate = Constraint(
        expr=m.fs.h2_production == (1000 / 54.517) * m.fs.np_to_electrolyzer,
        doc="Computes the hydrogen production rate"
    )
    # Tank holdup calculations (Assuming a delta_t of 1 hr)
    m.fs.tank_mass_balance = Constraint(
        expr=m.fs.tank_holdup - m.fs.tank_holdup_previous ==
             (m.fs.h2_production - m.fs.h2_to_pipeline - m.fs.h2_to_turbine)
    )

    # Compute the power production via h2 turbine
    # For an air_h2_ratio of 10.76, (T, P) of h2 = (300 K, 1 atm),
    # delta_p across compressor and turbine 24.1 bar, the conversion
    # factor is 0.0125 MW-hr/kg hydrogen
    m.fs.calc_turbine_power = Constraint(
        expr=m.fs.h2_turbine_power == 0.0125 * m.fs.h2_to_turbine,
        doc="Computes the power production via h2 turbine"
    )
    # Power balance at the grid
    m.fs.grid_power_balance = Constraint(
        expr=m.fs.net_power == m.fs.np_to_grid + m.fs.h2_turbine_power
    )

    return m


def build_scenario_model(ps):
    """
    ps: Object containing the parameters and set information
    """
    set_hours = ps.set_hours
    set_days = ps.set_days
    set_years = ps.set_years

    m = ConcreteModel()

    # Declare first-stage variables
    m.pem_capacity = Var(within=NonNegativeReals,
                         doc="Maximum capacity of the PEM electrolyzer (in MW)")

    m.tank_capacity = Var(within=NonNegativeReals,
                          doc="Maximum holdup of the tank (in kg)")
    m.h2_turbine_capacity = Var(within=NonNegativeReals,
                                doc="Maximum power output from the turbine (in MW)")

    # Append second-stage decisions and constraints
    m.period = Block(set_hours, set_days, set_years, rule=build_ne_flowsheet)

    # Tank holdup relations
    @m.Constraint(set_hours, set_days, set_years)
    def tank_holdup_relation(blk, t, d, y):
        if t == 1:
            # Pretending that the initial holdup is zero
            return (
                    blk.period[t, d, y].fs.tank_holdup_previous == 0
            )
        else:
            return (
                    blk.period[t, d, y].fs.tank_holdup_previous ==
                    blk.period[t - 1, d, y].fs.tank_holdup
            )

    @m.Constraint(set_hours, set_days, set_years)
    def pem_capacity_constraint(blk, t, d, y):
        return blk.period[t, d, y].fs.np_to_electrolyzer <= blk.pem_capacity

    @m.Constraint(set_hours, set_days, set_years)
    def tank_capacity_constraint(blk, t, d, y):
        return blk.period[t, d, y].fs.tank_holdup <= blk.tank_capacity

    @m.Constraint(set_hours, set_days, set_years)
    def turbine_capacity_constraint(blk, t, d, y):
        return blk.period[t, d, y].fs.h2_turbine_power <= blk.h2_turbine_capacity

    # H2 market
    @m.Constraint(set_hours, set_days, set_years)
    def h2_demand_constraint(blk, t, d, y):
        return blk.period[t, d, y].fs.h2_to_pipeline == ps.h2_demand

    return m


def app_costs_and_revenue(m, ps, scenario):
    """
    ps: Object containing information on sets and parameters
    """

    set_hours = ps.set_hours
    set_days = ps.set_days
    set_years = ps.set_years
    weights_days = ps.weights_days
    weights_years = ps.weights_years
    LMP = ps.LMP[scenario]

    h2_sp = ps.h2_price  # Selling price of hydrogen
    plant_life = ps.plant_life
    tax_rate = ps.tax_rate

    # We can remove the capex and fixed expressions
    # Turn the capex into individual capex cashflows
    # Note: LHV of hydrogen is 33.3 kWh/kg, Turbine capex fixed 1.06M
    m.capex = Expression(
            #https://www.hydrogen.energy.gov/pdfs/20004-cost-electrolytic-hydrogen-production.pdf
        expr=(1500000 * m.pem_capacity +
              #https: // www.energy.gov / eere / fuelcells / hydrogen-storage
              333 * m.tank_capacity +
            #https://www.pv-magazine.com/2021/07/14/hydrogen-fired-gas-turbines-vs-lithium-ion-storage/#:~:text=lithium%2Dion%20batteries%2C%20published%20in,network%20on%20a%20seasonal%20basis.
              1320000 * m.h2_turbine_capacity),
        doc="Total capital cost (in USD)"
    )

    #These will need to turn into individual recurring cash flows. They are fixed so based on size only.
    m.fixed_om_cost = Expression(
            #https://www.energy.gov/sites/prod/files/2014/08/f18/fcto_2014_electrolytic_h2_wkshp_colella1.pdf
        expr=((1500000*0.03) * m.pem_capacity +
            #https://www.pv-magazine.com/2021/07/14/hydrogen-fired-gas-turbines-vs-lithium-ion-storage/#:~:text=lithium%2Dion%20batteries%2C%20published%20in,network%20on%20a%20seasonal%20basis.
              (13000 * m.h2_turbine_capacity)),
        doc="Fixed O&M Cost (in USD)"
    )

    # H2 Credit
    @m.Expression(set_years)
    def h2_credit(blk, y):
        return (
                3 * sum(weights_days[y][d] * blk.period[t, d, y].fs.h2_production
                        for t in set_hours for d in set_days)
        ) #take out the 3 here and this is the function for the amount of hydrogen produced

    # Variable O&M: PEM: $1.3/MWh and turbine: $4.25/MWh
    @m.Expression(set_years)
    def variable_om_cost(blk, y):
        return (
                1.3 * sum(weights_days[y][d] * blk.period[t, d, y].fs.np_to_electrolyzer
                          for t in set_hours for d in set_days) +
                4.25 * sum(weights_days[y][d] * blk.period[t, d, y].fs.h2_turbine_power
                           for t in set_hours for d in set_days)
        )

    #This is a time-series I think? TEAL doesn't do time-series so this might be a challenge to implement
    @m.Expression(set_years)
    def electricity_revenue(blk, y):
        return (
            sum(weights_days[y][d] * LMP[y][d][t] * blk.period[t, d, y].fs.net_power
                for t in set_hours for d in set_days)
        )

    #remove h2_sp and it's the expression for the amount of H2 sold -- let's get rid of this and just consider the credit
    @m.Expression(set_years)
    def h2_revenue(blk, y):
        return (
                h2_sp * sum(weights_days[y][d] * blk.period[t, d, y].fs.h2_to_pipeline
                            for t in set_hours for d in set_days)
        )
    
    #I don't think we will need this expression since TEAL does MACRS depreciation
    @m.Expression(set_years)
    def depreciation(blk, y):
        return (
                blk.capex / plant_life
        )
    
    #don;t need this expression because TEAl calculates this
    @m.Expression(set_years)
    def net_profit(blk, y):
        return (
                blk.depreciation[y] + (1 - tax_rate) * (+ blk.h2_revenue[y]
                                                        + blk.electricity_revenue[y]
                                                        - blk.fixed_om_cost
                                                        - blk.variable_om_cost[y]
                                                        - blk.depreciation[y])
                + blk.h2_credit[y]
        )

    m.npv = Expression(
        expr=sum(weights_years[y] * m.net_profit[y] for y in set_years) - m.capex
    )



def build_stochastic_program(m):
    # Declare first-stage variables (Design decisions)
    m.pem_capacity = Var(within=NonNegativeReals,
                         doc="Maximum capacity of the PEM electrolyzer (in kW)")

    m.tank_capacity = Var(within=NonNegativeReals,
                          doc="Maximum holdup of the tank (in mol)")
    m.h2_turbine_capacity = Var(within=NonNegativeReals,
                                doc="Maximum power output from the turbine (in W)")

    # Build the model for one scenario
    sce_model = build_scenario_model(m)

    # Clone the model for all the scenarios
    m.scenarios = Block(m.set_scenarios)
    for s1 in m.set_scenarios:
        m.scenarios[s1].transfer_attributes_from(sce_model.clone())

        # Append cash flows for the scenario
        app_costs_and_revenue(m.scenarios[s1], m, scenario=s1)

    # Add non-anticipativity constraints
    @m.Constraint(m.set_scenarios)
    def non_anticipativity_pem(blk, s):
        return blk.pem_capacity == blk.scenarios[s].pem_capacity

    @m.Constraint(m.set_scenarios)
    def non_anticipativity_tank(blk, s):
        return blk.tank_capacity == blk.scenarios[s].tank_capacity

    @m.Constraint(m.set_scenarios)
    def non_anticipativity_turbine(blk, s):
        return blk.h2_turbine_capacity == blk.scenarios[s].h2_turbine_capacity


def append_objective_function(m):
    m.expectation_npv = Objective(
        expr=sum(m.weights_scenarios[s] * m.scenarios[s].npv for s in m.set_scenarios),
        sense=maximize
    )


if __name__ == '__main__':
    start = time.time()

    mdl = ConcreteModel()

    # Price of hydrogen: $2 per kg
    mdl.h2_price = 2


    # Flowrate of hydrogen to the pipeline in (kg/hr)
    mdl.h2_demand =10000
    #mdl.h2_demand = 0

    # Append LMP signal
    # append_lmp_signal(mdl,
    #                   signal_source="ARPA_E",
    #                   signal_name="MiNg_$100_CAISO")
    append_raven_lmp_signal(mdl,
                            scenarios=[0, 1],
                            years=[i for i in range(2020, 2040)],
                            plant_life=31,
                            discount_rate=0.09,
                            tax_rate=0.25)

    # Build the two-stage stochastic program
    build_stochastic_program(mdl)

    # Append the objective function
    append_objective_function(mdl)

    solver = SolverFactory("ipopt")
    solver.solve(mdl, tee=True)

    # full_year_plotting(mdl)

    # print("Revenue from electricity: $M ", mdl.electricity_revenue.expr() / 1e6)
    # print("Revenue from hydrogen   : $M ", mdl.h2_revenue.expr() / 1e6)
    # print("Revenue from credit     : $M ", mdl.h2_credit.expr() / 1e6)
    # print("Net profit              : $M ", mdl.net_profit.expr() / 1e6)
    # print("Total capital cost      : $M ", mdl.capex.expr() / 1e6)
    print("Net present value       : $M ", mdl.expectation_npv.expr() / 1e6)
    print()

    # print("Minimum PEM Capacity    : ", 54.517 * 1e-3 * mdl.h2_demand, " MW")
    print("PEM Capacity            : ", mdl.pem_capacity.value, " MW")
    print("Tank Capacity           : ", mdl.tank_capacity.value, " kg")
    print("H2 Turbine Capacity     : ", mdl.h2_turbine_capacity.value, " MW")
    print("Electricity Revenue       : ", mdl.scenarios[0].electricity_revenue[2021].expr())
    print("H2 Credit Revenue       : ", mdl.scenarios[0].h2_credit[2021].expr())

    end = time.time()
    print(f"Time taken for the run: {end - start} s")
