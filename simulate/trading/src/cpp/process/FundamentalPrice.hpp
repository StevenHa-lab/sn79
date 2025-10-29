/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "taosim/simulation/ISimulation.hpp"
#include "Process.hpp"
#include "RNG.hpp"
#include "common.hpp"
#include <Eigen/Dense>


#include <pugixml.hpp>

//-------------------------------------------------------------------------

class FundamentalPrice : public Process
{
public:
    FundamentalPrice(
        taosim::simulation::ISimulation* simulation,
        uint64_t bookId,
        uint64_t seedInterval,
        double mu,
        double sigma,
        double dt,
        double X0,
        double lambda,
        double muJump,
        double sigmaJump,
        Timestamp updatePeriod,
        double hurst= 0.5,
        double epsilon=0.0) noexcept;

    virtual void update(Timestamp timestamp) override;
    virtual double value() const override;
    virtual void checkpointSerialize(
        rapidjson::Document& json, const std::string& key = {}) const override;

    [[nodiscard]] static std::unique_ptr<FundamentalPrice> fromXML(
        taosim::simulation::ISimulation* simulation, pugi::xml_node node, uint64_t bookId, double X0);
    [[nodiscard]] static std::unique_ptr<FundamentalPrice> fromCheckpoint(
        taosim::simulation::ISimulation* simulation, const rapidjson::Value& json, double X0);

private:
    taosim::simulation::ISimulation* m_simulation;
    uint64_t m_bookId;
    uint64_t m_seedInterval;
    std::string m_seedfile;
    RNG m_rng;
    double m_X0, m_mu, m_sigma, m_dt;
    double m_dJ;
    double m_t{}, m_W{};
    std::normal_distribution<double> m_gaussian;
    // fBm
    Eigen::MatrixXd m_L; 
    Eigen::VectorXd m_X;
    Eigen::VectorXd m_V;
    double m_epsilon;
    double m_hurst;
    std::normal_distribution<double> m_fractional_gaussian;
    double m_BH{};
    void cholesky_step(int64_t i);
    double gamma_fn(int64_t k, double H) const;
    //end
    std::normal_distribution<double> m_jump;
    std::poisson_distribution<int> m_poisson;
    double m_value;
    int m_last_count = 0;
    uint64_t m_last_seed = 0;
    Timestamp m_last_seed_time = 0;
};

//-------------------------------------------------------------------------