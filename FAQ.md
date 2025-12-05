<div align="center">

# **τaos** ☯ **‪ي‬n 79**<!-- omit in toc -->
### **Decentralized Simulation of Automated Trading in Intelligent Markets:** <!-- omit in toc -->
### **Risk-Averse Agent Optimization** <!-- omit in toc -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
---
# Frequently Asked Questions

</div>

#### 1. How does τaos differ from other finance-related subnets?

Other finance-related subnets, at least to our knowledge at time of writing, focus on incentivizing the creation and deployment of trading strategies which act against particular real-world markets, and seek to extract value from the trading signals produced by miners.  While this approach has some promise, τaos has more general aspirations to provide value across a broad spectrum of use cases within the financial industry.  By providing an environment where miners trade in many statistically similar but independently evolving simulated markets simultaneously, we not only encourage the study and development of much more robust trading strategies, but also produce vast quantities of high-resolution, maximally detailed data which can be used by traders, researchers, institutions and regulators to better understand and account for the underlying risks present in all markets.

#### 2. How are miners evaluated?

While the exact details of the incentive mechanism are subject to change over time, the key objective for miners is always to maximize average _risk-adjusted_ performance over time across all simulated orderbooks.  The central metric applied in measuring performance is the Sharpe ratio, a very well known and established tool in evaluating trading success.  The returns for the Sharpe calculation are defined as the difference in total estimated inventory value between observations, and the measurement is taken over a rolling window of length defined in terms of the number of observations in the [validator config](/taos/im/config/__init__.py) as `--scoring.sharpe.lookback`.  Note that the actual simulation time period of the assessment is related to this through the `Simulation.step` in the [simulation config](/simulate/trading/run/config/simulation_0.xml) - a new observation is obtained every `Simulation.step` simulation nanoseconds.  

There are then a few additional transformations applied to avoid manipulation and encourage active trading; miners' scores are scaled in proportion to their trading volume over the Sharpe assessment window, with a decay factor being applied during any inactive periods.  A penalty is applied to miners scores in cases where significant outliers in terms of performance exist between the simulated books.  The full details of the implementation can be understood by studying the [reward logic](/taos/im/validator/reward.py).  The overall score of a miner is determined as an exponential moving average of the score calculated at each observation, with the period of the EMA being set via the `--neuron.moving_average_alpha` parameter which is applied in the [base validator logic](/taos/dev/taos/common/neurons/validator.py).

#### 3. How do I get started mining in the subnet?

This FAQ is a good entry point, after which it is recommended to go through the [README](/README.md).  Once you are familiar with the subnet function and vision, and decide you want to get involved, you can check out the [agents readme](/taos/dev/agents/README.md) for more detailed information on how to design and develop strategies for the τaos framework.  Before getting into mining on testnet or mainnet, we recommend also to set up a local testing environment using the ["proxy" validator](/taos/dev/agents/proxy/README.md) tools which enable to launch a local instance of the simulation engine and confirm how your strategy behaves and performs against the background market.

Once you have developed and tested your strategy locally, you may next wish to inspect the current behaviour and performance of existing miners in the subnet via the [taos.simulate.trading dashboard](https://taos.simulate.trading) (an updated version of this dashboard as well as documentation to assist in interpreting the visualizations are upcoming).To get an idea of the expected performance of your strategy and validate your hosting and networking configuration, you can request testnet TAO via the [Bittensor Discord](https://discord.com/channels/799672011265015819/1389370202327748629), register a UID in our test netuid 366 and deploy and [monitor](https://testnet.simulate.trading) your miner here.  We continously run a validator in testnet using the latest code and configurations.  Once you are confident that your agent has what it takes, register to mainnet and join the comptetition!  If you have any questions or concerns, reach out to us at our [Discord Channel](https://discord.com/channels/799672011265015819/1353733356470276096).

**Note that the example miners given in the /agents directory are not expected to perform well in the subnet - you need to develop a smart custom algorithm to compete**

#### 4. Is there a τaos testnet?

Yes, netuid 366 with monitoring via [testnet.simulate.trading](https://testnet.simulate.trading).

#### 5. How do I monitor my miner?

Data reflecting the current state of the simulation, including miner performance and activity, is published by validators at [taos.simulate.trading](https://taos.simulate.trading).  You can access detailed information for your specific miner by clicking on your UID in the "Agent" column of the "Agents" table - this view reports detailed metrics related to your agent's behaviour and performance.  An identical dashboard for testnet is available at [testnet.simulate.trading](https://testnet.simulate.trading).

#### 6. What is the update schedule and procedure for the subnet?

We target to release an update every week on Wednesday.  The content of the update will be announced ahead of time, usually earlier in the week or latest earlier in the day on Wednesday, with code being pushed to repo and deployed to testnet around 15:00 UTC.  Assuming no issues, the new version is deployed to mainnet at around 17:00 UTC.

In most cases the updates do not require changes from miners, in situations where the code changes require miners to update this will be clearly communicated.  Changes having a wider impact will be announced further ahead of time and run on testnet for an extended period to ensure a smooth deployment to live.

#### 7. What is "simulation time" and why is it different from real world time?

Being that our markets are synthetic and generated through a powerful C++ engine which creates many statistically realistic limit order books simultaneously, the simulation maintains an internal "clock" which tracks to nanosecond precision the time elapsed since the start of simulation in a manner consistent with the evolution.  Due mostly to constraints related to the requirement to query miners with the latest state information for all books and await responses at regular intervals, the time in the simulation typically progresses more slowly than actual time - that is, over 1 hour of real time, the simulation may only progress by 20 minutes (though it should be kept in mind that this represents the evolution of many simulated books over a 20 minute period).  We work to reduce this discrepancy and decrease time taken in the query process, while we have already started planning the next iteration of the subnet which would enable real-time communication between miners and validators, eliminating this query process delay entirely.

#### 8. My miner seems to be receiving requests and responding, but I don't see any activity and my score is not increasing.  What's going on?

If you have just registered the miner, note that scores are not assigned until sufficient time has passed to allow calculating a meaningful Sharpe ratio.  If the situation persists, you will need to check your UID at the [Agents Dashboard](https://taos.simulate.trading/d/edy6vxytuud4wd/agents) and confirm a few critical things:

- Do you see recent trades for all book IDs?  Miners must trade on every book in order to receive score.
- Under the Requests plot, do you see a large proportion of failures or timeouts?  If you are not seeing mostly success, usually this is due to taking too long to respond - validators allow a maximum of `--neuron.timeout` seconds (defined in the [base validator config](/taos/dev/taos/common/config/__init__.py)) for miners to respond.  This can be addressed by increasing resources, optimizing your strategy logic and ensuring sufficient network connectivity; you may also want to consider geolocating your miner nearby to the biggest validators for the best possible latency.
- Pay attention to the Sharpe Score and Sharpe Penalty, these are the primary metrics used in determining miner score.

#### 9. As a miner, I've hit the trading volume limit and can no longer submit instructions.  How is this limit enforced and what can I do now?

In order to prevent miners from attempting to exploit the volume-weighting of scores and overloading the simulations with excessive careless trading activity, a "cap" is enforced on the total QUOTE volume allowed to be traded in a given period of simulation time.  Trading volume is calculated in QUOTE as the sum of price multiplied by quantity over all trades in which the miner is involved.  The period over which the trading volume limit is assessed is defined in the [validator config](/taos/im/config/__init__.py) as `--scoring.activity.trade_volume_assessment_period` (specified in simulation nanoseconds), where this is checked every `--scoring.activity.trade_volume_sampling_interval` (simulation nanoseconds) against the limit which is calculated as `--scoring.activity.capital_turnover_cap` multiplied by the initial wealth allocated to miners defined in the [simulation config](/simulate/trading/run/config/simulation_0.xml) as `Simulation.Agents.MultiBookExchangeAgent.Balances.wealth`.  

If a miner trades more than `Simulation.Agents.MultiBookExchangeAgent.Balances.wealth` * `--scoring.activity.capital_turnover_cap` in a period of `--scoring.activity.trade_volume_assessment_period` simulation nanoseconds, no more instructions will be accepted on that book (except cancellations) until the total QUOTE volume traded in the most recent `trade_volume_assessment_period` drops below the limit.  Miners need to consider this limitation when designing and testing strategies in order to maximize volume and Sharpe ratio without exceeding the limit.  If the cap is hit early in the first 24 hours, there is a high risk of deregistration as no further actions will be possible until at least 24 simulation hours have elapsed.  Your current total traded volume is included in the state update for easy reference, accessible in code via `self.accounts[book_id]['traded_volume]`.

Note also that the trading volumes used in the assessment are not reset when a new simulation begins; the trading volumes are determined based on a 24 hour period which may span multiple simulations.

#### 10. Why do validators in the subnet exhibit discrepancies in vTrust?

While we have made changes to attempt to better align the weights assigned to miners by different validators, the nature of the subnet operation does require that there exist a group of miners which consistently outperform others across all simulated markets in order that validators would agree on the top scoring UIDs.  The discrepancies result from a combination of inconsistent performance by miners, networking considerations leading to different success rates and latencies between miners and validators, and the fact that the simulation hosted by each validator is different from others due to the stochastic nature of the background model.  We continue to seek ways to improve this situation, but have to consider also the impact of these changes in relation to the utility of the subnet - if all validators are evaluating the same exact conditions, this eliminates a key element guaranteeing the robustness of the top scoring miners to perform well in all environments.

#### 11. Do you currently burn any miner emissions, or have any plans to implement this mechanism?

No, we do not burn miner emissions and do not currently have any plans to implement this.  Though this seems it may make sense in some other subnets, we do not see that this would be the case for us.  If in future we see need to apply such, this will not be done without careful consideration and consultation with all participants.

#### 12. Why does the current scoring system seem to favor passive market making over active trading?

The present scoring methodology relies on Sharpe ratios calculated from changes in inventory value over time. Due to this, miners who hold small and stable positions with limited activity can end up looking disproportionately strong. Standing limit orders that never get filled do not incur any trading costs such as fees or spreads, so these miners avoid many of the frictions faced by active traders. When inventory fluctuates only slightly, the resulting volatility is also low, which naturally produces a stable Sharpe ratio. Although the system includes volume weighting to promote meaningful activity, the Sharpe calculation itself does not always reflect the true quality of execution.

We are actively working on refinements that better differentiate genuine liquidity provision from passive “standing still” behavior. These improvements focus on execution-centered metrics that factor in trading costs, risk-adjusted performance based on actual fills, and overall operational efficiency.

#### 13. How does the scoring system account for trading costs like fees and spreads?

The Sharpe ratio is based on inventory value changes, and these changes implicitly include the costs or rebates associated with trading. When a miner pays fees or receives rebates, the resulting balance adjustments are captured in the inventory path, which then flows into the performance calculation. In addition, the [Dynamic Incentive Structure](https://simulate.trading/taos-im-dis-paper) (DIS) used in the simulator is specifically designed to amplify the impact of execution costs and to guide agents toward providing or removing liquidity in ways that match the prevailing market environment.  Although trading costs/rebates are incorporated in this way, the effect is diluted for large trades - this will be addressed in the context of including realized profitability to the scoring assessment.

#### 14. Does the scoring system penalize order cancellations or repeated re-posting?

Currently, the scoring framework does not directly penalize order cancellations or the repeated submission of identical orders, even when those actions do not result in meaningful changes. The system focuses more heavily on trading volume and risk-adjusted returns. However, we recognize that excessive cancellations or unchanging re-posts can place unnecessary load on the simulation, may represent score-farming rather than legitimate market-making behavior, and ultimately do not contribute to price discovery or improved liquidity.  Future refinements may apply small penalties or filters to discourage unnecessary churn while ensuring that traders responding appropriately to real market changes are not penalized.

Future versions of the scoring model are likely to introduce operational-efficiency considerations, such as penalties for high cancel-to-fill ratios or for repeatedly submitting identical orders that do not produce new placements or executions. We may also consider simpler guardrails at the agent level—for example, raising minimum order sizes, reducing the number of instructions permitted per round, or further limiting the maximum number of open orders.

#### 15. How does the scoring system encourage participation across all books?

The current system enforces a need to maintain trading activity on all books through the activity factors and associated decays, and further uses an outlier penalty that reduces a miner’s score when performance diverges significantly across books. While this provides some incentive for balanced participation, it does not explicitly reward breadth of coverage compared to specialization. To address this, we are exploring mechanisms that encourage miners to trade actively and intelligently on all books. These could include soft bonuses or penalties based on coverage, diminishing returns for extreme concentration in a small number of books, or the use of concentration metrics to detect overly narrow focus. Coverage incentives will be tied to executed value, not just order placements.

The goal is to support robust strategies that can adapt across diverse markets without artificially forcing traders to participate in books where they have no strategic advantage.

#### 16. Will the scoring system move toward execution-based metrics rather than inventory-based metrics?

We are actively considering a shift toward execution-based scoring which incorporates realized profitability into the calculation. The current inventory-delta Sharpe method has several limitations: it can reward miners for stable inventory positions even when they are not trading meaningfully and it struggles to capture the quality of execution in terms of timing and pricing.

A more execution-focused approach would reward miners more explicitly on the profits realized through "round-tripped" trades, in addition to their total inventory value. It would also emphasize net alpha after accounting for all trading costs, incorporate downside-risk measures such as CVaR instead of relying solely on standard deviation, and potentially incorporate measures like time-to-fill to reward efficient execution.  Operational penalties will be based on recorded cancels and reposts. This allows the system to encourage balanced execution while also discouraging inefficient operational churn.

#### 17. Why are some miners able to maintain high scores with minimal trading activity?

This effect arises when miners maintain extremely stable inventories with little volatility. Even without substantial trading volume, this stability can generate strong Sharpe ratios. The system counterbalances this to some extent through volume weights and by decaying scores during periods of inactivity, but finding the ideal balance is challenging. We do not want to force unnecessary trades, since overtrading can be harmful in real markets, yet we also want to prevent miners from achieving strong scores simply by avoiding risk.

Upcoming refinements aim to ensure that positive scoring requires genuine execution, incorporates all trading costs including spreads and slippage, and discourages strategies that achieve stability through inactivity instead of skillful risk management.

#### 18. Are identical re-posts treated as no-ops in the simulator?

No. Identical re-posts are fully recorded as cancel/placement events. They do not generate new fills unless market conditions change, but they still consume simulator resources and count toward operational activity. Any future scoring penalties aimed at discouraging churn will rely on these recorded events. Excessive cancel/repost cycles often add load without improving liquidity or execution quality - future scoring revisions may apply modest penalties based on recorded operational counts (e.g., cancel-to-fill ratios, repeated identical reposts) to discourage wasteful operational behavior while still allowing legitimate quote updates.

#### 19. How will a cost-aware, execution-focused scoring model treat taker trades?

Taker trades will be evaluated by their realized alpha after accounting for fees and half-spread costs in the same way as for maker trades. Since each taker fill and its costs are recorded, miners can be rewarded when taking liquidity is strategically correct and net-positive, while still internalizing the full cost of doing so.  The DIS already rewards/penalizes taking in conditions where it is less appropriate; this is intended to be further refined and augmented via the incentive mechanism to better encourage liquidity taking and active trading when this would be genuinely beneficial in improving market quality.

#### 20. How will scoring handle time-weighted quoting quality (e.g., being near the top of book)?

Execution metrics can fail to reward traders who continuously post competitive quotes which contribute to market quality, but see low fill rates due to the dynamics of the market. We plan to explore time-weighted measures of quote quality, such as time spent quoting within a certain percentage of the best bid/ask, to complement execution- and inventory-based metrics. These would not replace fill-based scoring but could serve as a soft multiplier ensuring that high-quality liquidity provision is not undervalued in quiet markets.

#### 21. How will the system ensure miners don’t just optimize for scoring rather than real liquidity?

The design goal is to align good scoring outcomes with behaviors that genuinely improve market quality. This alignment will be strengthened through execution-focused metrics, realistic cost accounting, operational-efficiency considerations, and coverage incentives, all of which reduce gaming opportunities and encourage traders to trade in a way that promotes liquidity and healthy market dynamics.