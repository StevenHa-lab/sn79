/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "taosim/message/ExchangeAgentMessagePayloads.hpp"
#include "taosim/message/MessagePayload.hpp"
#include "Distribution.hpp"
#include "Agent.hpp"
#include "GBMValuationModel.hpp"
#include "Order.hpp"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/math/distributions/rayleigh.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <numeric>
#include <random>

//-------------------------------------------------------------------------

class HighFrequencyTraderAgent : public Agent
{
public:
    HighFrequencyTraderAgent(Simulation* simulation) noexcept;

    virtual void configure(const pugi::xml_node& node) override;
    virtual void receiveMessage(Message::Ptr msg) override;

private:
    struct TopLevel
    {
        double bid, ask;
        double bidQty, askQty;
    };

    struct DelayBounds
    {
        Timestamp min, max;
    };

    struct RecordedOrder {
        OrderID orderId;
        double price;
        double volume;
        OrderDirection direction;
        bool traded;
        bool canceled;
    };

    struct TimestampedTradePrice
    {
        Timestamp timestamp{};
        double price{};
    };
    
    void handleSimulationStart();
    void handleSimulationStop();
    void handleTradeSubscriptionResponse();
    void handleRetrieveL1Response(Message::Ptr msg);
    void handleLimitOrderPlacementResponse(Message::Ptr msg);
    void handleLimitOrderPlacementErrorResponse(Message::Ptr msg);
    void handleMarketOrderPlacementResponse(Message::Ptr msg);
    void handleMarketOrderPlacementErrorResponse(Message::Ptr msg);
    void handleCancelOrdersResponse(Message::Ptr msg);
    void handleCancelOrdersErrorResponse(Message::Ptr msg);
    void handleTrade(Message::Ptr msg);

    void placeOrder(BookId bookId, TopLevel& topLevel);
    std::optional<PlaceOrderLimitPayload::Ptr> makeOrder(BookId bookId, OrderDirection direction, 
        double volume, double limitPrice, double wealth);
    void sendOrder(std::optional<PlaceOrderLimitPayload::Ptr> payload);
    void cancelClosestToBestPrice(BookId bookId, OrderDirection direction, double bestPrice);
    Timestamp orderPlacementLatency();
    void recordOrder(PlaceOrderLimitResponsePayload::Ptr payload);
    void removeOrder(BookId bookId, OrderID orderId, std::optional<double> amount = {});


    std::mt19937 *m_rng;
    std::string m_exchange;
    uint32_t m_bookCount;

    double m_wealthFrac;
    double m_priceInit;
    double m_gHFT;
    double m_kappa;
    double m_spread;
    double m_delta;
    double m_tau;
    Timestamp m_minMFLatency;
    double m_psi;
    
    double m_noiseRay;
    double m_shiftPercentage;
    std::vector<double> m_orderMean;
    double m_orderSTD;

    double m_pRes;
    double m_sigmaSqr;
    DelayBounds m_opl;
    double m_targetMTR;
    double m_margin;
    double m_sigmaMargin;
    double m_rateSensitivity;
    double m_spreadSensitivityExp;
    double m_spreadSensitivityBase;
    
    uint32_t m_priceDecimals;
    uint32_t m_volumeDecimals;
    double m_priceIncrement;
    double m_volumeIncrement;
    double m_maxLoan;
    double m_maxLeverage;
    bool m_debug;
    AgentId m_id;

    std::vector<TopLevel> m_topLevel;
    std::vector<double> m_inventory;
    std::vector<double> m_baseFree;
    std::vector<double> m_quoteFree;
    std::vector<bool> m_orderFlag;
    std::map<BookId, std::vector<RecordedOrder>> m_recordedOrders;

    std::vector<double> m_deltaHFT;
    std::vector<Timestamp> m_tauHFT;

    std::vector<boost::circular_buffer<double>> m_priceHist;
    std::vector<boost::circular_buffer<double>> m_logReturns;
    std::vector<TimestampedTradePrice> m_tradePrice;
    std::unique_ptr<taosim::stats::Distribution> m_orderPlacementLatencyDistribution;
    std::unique_ptr<taosim::stats::Distribution> m_priceShiftDistribution;
    std::normal_distribution<double> m_marginDist;
};

//-------------------------------------------------------------------------
