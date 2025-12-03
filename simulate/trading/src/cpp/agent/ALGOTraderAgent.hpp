/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "Agent.hpp"
#include "Distribution.hpp"
#include "Order.hpp"
#include "Trade.hpp"
#include "taosim/message/ExchangeAgentMessagePayloads.hpp"
#include "taosim/decimal/decimal.hpp"


#include <memory>
#include <queue>
#include <random>

//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------

enum class ALGOTraderStatus : uint32_t
{
    ASLEEP,
    READY,
    EXECUTING
};

struct ALGOTraderExecutionInfo
{
    OrderDirection dir;
    decimal_t volumeToBeExecuted;
};

struct TimestampedVolume
{
    Timestamp timestamp;
    decimal_t volume;
    decimal_t price; // VWAP for logrets

    [[nodiscard]] auto operator<=>(const TimestampedVolume& other) const noexcept
    {
        return timestamp <=> other.timestamp;
    }
};

struct BookStat 
{
    double bid,ask;
};

class ALGOTraderVolumeStats
{
public:
    explicit ALGOTraderVolumeStats(Timestamp period, 
        double alpha, double beta, double omega, double gamma, 
        double initPrice, size_t depth);

    void push(const Trade& trade);
    void push(TimestampedVolume timestampedVolume);
    void push_levels(Timestamp timestamp, std::vector<BookLevel>& bids, std::vector<BookLevel>& asks);
    
    [[nodiscard]] decimal_t rollingSum() const noexcept { return m_rollingSum; }
    [[nodiscard]] double variance() const noexcept { return m_variance; }
    [[nodiscard]] double estimatedVolatility() const noexcept { return std::pow(m_estimatedVol, 0.5)* std::pow(86'400'000'000'000/m_period ,0.5); }
    [[nodiscard]] double bidSlope() noexcept { return lastSlopes().bid; }
    [[nodiscard]] double askSlope() noexcept { return lastSlopes().ask; }
    [[nodiscard]] double bidVolume() noexcept { return lastVolume().bid; }
    [[nodiscard]] double askVolume() noexcept { return lastVolume().ask; }

    [[nodiscard]] static ALGOTraderVolumeStats fromXML(pugi::xml_node node, double initPrice, size_t depth);

private:
    double slopeOLS(std::vector<BookLevel>& side);
    double volumeSum(std::vector<BookLevel>& side, size_t depth = 5);
    BookStat lastSlopes() {return m_bookSlopes[m_lastSeq]; }
    BookStat lastVolume() {return m_bookVolumes[m_lastSeq]; }

    
    Timestamp m_period;
    double m_alpha;
    double m_beta;
    double m_omega;
    double m_gamma;
    double m_initPrice;
    size_t m_depth;

    std::priority_queue<
        TimestampedVolume,
        std::vector<TimestampedVolume>,
        std::greater<TimestampedVolume>> m_queue;
    decimal_t m_rollingSum{};
    std::map<Timestamp,double> m_priceHistory; // Timestamped price history, close price (VWAP per exact timestamp)
    std::map<Timestamp,double> m_logRets; 
    double m_priceLast;
    double m_variance;
    double m_estimatedVol;
    Timestamp m_lastSeq;
    std::map<Timestamp, BookStat> m_bookSlopes; 
    std::map<Timestamp, BookStat> m_bookVolumes;
    
};

struct ALGOTraderState
{
    ALGOTraderStatus status;
    Timestamp marketFeedLatency;
    ALGOTraderVolumeStats volumeStats;
    decimal_t volumeToBeExecuted;
    OrderDirection direction;
    Timestamp statusChangeTime;
    Timestamp statusChangeEndTime;
};

//-------------------------------------------------------------------------

class ALGOTraderAgent : public Agent
{
public:
    explicit ALGOTraderAgent(Simulation* simulation) noexcept;

    virtual void configure(const pugi::xml_node& node) override;
    virtual void receiveMessage(Message::Ptr msg) override;

private:
    struct TopLevel
    {
        double bid, ask;
    };
    struct DelayBounds
    {
        Timestamp min, max;
    };

    struct VolatilityBounds
    {
        double activationRate, activationMidpoint, activationCapacity;
    };

    void handleSimulationStart(Message::Ptr msg);
    void handleTrade(Message::Ptr msg);
    void handleWakeup(Message::Ptr msg);
    void handleMarketOrderResponse(Message::Ptr msg);
    void handleBookResponse(Message::Ptr msg);
    void handleL1Response(Message::Ptr msg);

    void execute(BookId bookId, ALGOTraderState& state);
    decimal_t drawNewVolume(uint32_t baseDecimals);
    double getProcessValue(BookId bookId, const std::string& name);
    uint64_t getProcessCount(BookId bookId, const std::string& name);
    double wakeupProb(ALGOTraderState& state, double fundDist);
    Timestamp orderPlacementLatency();
    Timestamp marketFeedLatency();

    std::mt19937* m_rng;
    std::string m_exchange;
    uint32_t m_bookCount;
    double m_volumeMin;
    std::unique_ptr<stats::Distribution> m_volumeDistribution;
    std::vector<ALGOTraderState> m_state;
    double m_opLatencyScaleRay;
    DelayBounds m_opl;
    std::normal_distribution<double> m_marketFeedLatencyDistribution;
    std::unique_ptr<stats::Distribution> m_orderPlacementLatencyDistribution;
    std::unique_ptr<stats::Distribution> m_volumeDrawDistribution;
    std::vector<decimal_t> m_lastPrice;
    std::vector<TopLevel> m_topLevel;
    std::normal_distribution<double> m_departureThreshold;
    VolatilityBounds m_volatilityBounds;
    double m_durationProbCoef, m_deviationProbCoef;
    Timestamp m_period;
    size_t m_depth;
    std::normal_distribution<double> m_delay;
    double m_immediateBase;
   
    u_int64_t m_lastTriggerUpdate;
};

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
