# tia_project

Repository with the aim of carrying out the practical work for the "TIA" subject of the MUIARFID program at UPV during the 2023/2024 academic year.

## The problem

The problem is based on the energy generation by various generators for a consumer. There are n time slots in which the generators must meet the consumer's demand. Each generator has a maximum generation value per slot, as well as the cost per KW and whether the energy they produce is renewable or not. Additionally, generators cannot be turned on for more than m (m < n) time slots, and for those slots in which they are turned on, we must account for a fixed operating cost associated with their operation.

The consumer will pay the cost of the most expensive energy used in the generation. If more energy is supplied than needed, they will only pay for what was required. Each generator will contribute a percentage of the maximum generation it can produce. The objectives are to maximize the profit of the generators and to reduce the energy generation by non-renewable energy generators.

## The solution

To solve this problem, different approaches will be used to model and address it. The first approach will be through genetic algorithms, and the second one will be based on simulated annealing.