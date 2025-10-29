# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

if __name__ != "__mp_main__":
    import os
    import json
    import platform
    import time
    import argparse
    import torch
    import traceback
    import xml.etree.ElementTree as ET
    import pandas as pd
    import msgspec
    import math
    import shutil
    import zipfile
    import asyncio
    import posix_ipc
    import mmap
    import msgpack
    from datetime import datetime, timedelta
    from ypyjson import YpyObject

    import bittensor as bt

    import uvicorn
    from typing import Tuple
    from fastapi import FastAPI, APIRouter
    from fastapi import Request
    from threading import Thread

    import subprocess
    import psutil
    from git import Repo
    from pathlib import Path

    from taos import __spec_version__
    from taos.common.neurons.validator import BaseValidatorNeuron
    from taos.im.utils import duration_from_timestamp

    from taos.im.config import add_im_validator_args
    from taos.im.protocol.simulator import SimulatorResponseBatch
    from taos.im.protocol import MarketSimulationStateUpdate, FinanceEventNotification
    from taos.im.protocol.models import MarketSimulationConfig, TradeInfo
    from taos.im.protocol.events import SimulationStartEvent, TradeEvent

    class Validator(BaseValidatorNeuron):
        """
        intelligent market simulation validator implementation.

        The validator is run as a FastAPI client in order to receive messages from the simulator engine for processing and forwarding to miners.
        Metagraph maintenance, weight setting, state persistence and other general bittensor routines are executed in a separate thread.
        The validator also handles publishing of metrics via Prometheus for visualization and analysis, as well as retrieval and recording of seed data for simulation price process generation.
        """

        @classmethod
        def add_args(cls, parser: argparse.ArgumentParser) -> None:
            """
            Add intelligent-markets-specific validator configuration arguments.
            """
            add_im_validator_args(cls, parser)

        def _maintain(self) -> None:
            try:
                self.maintaining = True
                bt.logging.info(f"Synchronizing at Step {self.step}...")
                start=time.time()
                self.sync(save_state=False)
                if not check_simulator(self):
                    restart_simulator(self)
                bt.logging.info(f"Synchronized ({time.time()-start:.4f}s)")
            except Exception as ex:
                self.pagerduty_alert(f"Failed to sync : {ex}", details={"trace" : traceback.format_exc()})
            finally:
                self.maintaining = False

        def maintain(self) -> None:
            """
            Maintains the metagraph and sets weights.
            This method is run in a separate thread to speed up processing.
            """
            if not self.maintaining:
                Thread(target=self._maintain, args=(), daemon=True, name=f'maintain_{self.step}').start()

        def monitor(self) -> None:
            while True:
                try:
                    time.sleep(300)
                    bt.logging.info(f"Checking simulator state...")
                    if not check_simulator(self):
                        restart_simulator(self)
                    else:
                        bt.logging.info(f"Simulator online!")
                except Exception as ex:
                    bt.logging.error(f"Failure in simulator monitor : {traceback.format_exc()}")

        def seed(self) -> None:
            from taos.im.validator.seed import seed
            seed(self)

        def update_repo(self, end=False) -> bool:
            """
            Checks for and pulls latest changes from the taos repo.
            If source has changed, the simulator is rebuilt and restarted.
            If config has changed, restart the simulator to activate the new parameterizations.
            If validator source is updated, restart validator process.
            """
            try:
                validator_py_files_changed, simulator_config_changed, simulator_py_files_changed, simulator_cpp_files_changed = check_repo(self)
                remote = self.repo.remotes[self.config.repo.remote]

                if not end:
                    if validator_py_files_changed and not (simulator_cpp_files_changed or simulator_py_files_changed):
                        bt.logging.warning("VALIDATOR LOGIC UPDATED - PULLING AND DEPLOYING.")
                        remote.pull()
                        update_validator(self)
                else:
                    try:
                        remote.pull()
                    except Exception as ex:
                        self.pagerduty_alert(f"Failed to pull changes from repo on simulation end : {ex}")
                    if simulator_cpp_files_changed or simulator_py_files_changed:
                        bt.logging.warning("SIMULATOR SOURCE CHANGED")
                        rebuild_simulator(self)
                    if simulator_config_changed:
                        bt.logging.warning("SIMULATOR CONFIG CHANGED")
                    restart_simulator(self)
                    if validator_py_files_changed:
                        update_validator(self)
                return True
            except Exception as ex:
                self.pagerduty_alert(f"Failed to update repo : {ex}", details={"traceback" : traceback.format_exc()})
                return False

        def _compress_outputs(self,  start=False):
            self.compressing = True
            try:
                if self.simulation.logDir:
                    log_root = Path(self.simulation.logDir).parent
                    for output_dir in log_root.iterdir():
                        if output_dir.is_dir():
                            log_archives = {}
                            log_path = Path(output_dir)
                            for log_file in log_path.iterdir():
                                if log_file.is_file() and log_file.suffix == '.log':
                                    log_period = log_file.name.split('.')[1]
                                    if len(log_period) == 13:
                                        log_end = (int(log_period.split('-')[1][:2]) * 3600 + int(log_period.split('-')[1][2:4]) * 60 + int(log_period.split('-')[1][4:])) * 1_000_000_000
                                    else:
                                        log_end = (int(log_period.split('-')[1][:2]) * 86400 + int(log_period.split('-')[1][2:4]) * 3600 + int(log_period.split('-')[1][4:6]) * 60 + int(log_period.split('-')[1][6:])) * 1_000_000_000
                                    if log_end < self.simulation_timestamp or (start and str(output_dir.resolve()) != self.simulation.logDir):
                                        log_type = log_file.name.split('-')[0]
                                        label = f"{log_type}_{log_period}"
                                        if not label in log_archives:
                                            log_archives[label] = []
                                        log_archives[label].append(log_file)
                            for label, log_files in log_archives.items():
                                archive = log_path / f"{label}.zip"
                                bt.logging.info(f"Compressing {label} files to {archive.name}...")
                                with zipfile.ZipFile(archive, "w" if not archive.exists() else "a", compression=zipfile.ZIP_DEFLATED) as zipf:
                                    for log_file in log_files:
                                        try:
                                            zipf.write(log_file, arcname=Path(log_file).name)
                                            os.remove(log_file)
                                            bt.logging.debug(f"Added {log_file.name} to {archive.name}")
                                        except Exception as ex:
                                            bt.logging.error(f"Failed to add {log_file.name} to {archive.name} : {ex}")
                    if psutil.disk_usage('/').percent > 85:
                        min_retention_date = int((datetime.today() - timedelta(days=7)).strftime("%Y%m%d"))
                        bt.logging.warning(f"Disk usage > 85% - cleaning up old outputs...")
                        for output in sorted(log_root.iterdir(), key=lambda f: f.name[:13]):
                            try:
                                archive_date = int(output.name[:8])
                            except:
                                continue
                            if archive_date < min_retention_date:
                                try:
                                    if output.is_file() and output.name.endswith('.zip'):
                                        output.unlink()
                                    elif output.is_dir():
                                        shutil.rmtree(output)
                                    disk_usage = psutil.disk_usage('/').percent
                                    bt.logging.success(f"Deleted {output.name} ({disk_usage}% disk available).")
                                    if disk_usage <= 85:
                                        break
                                except Exception as ex:
                                    self.pagerduty_alert(f"Failed to remove output {output.name} : {ex}", details={"trace" : traceback.format_exc()})


            except Exception as ex:
                self.pagerduty_alert(f"Failure during output compression : {ex}", details={"trace" : traceback.format_exc()})
            finally:
                self.compressing = False

        def compress_outputs(self, start=False):
            if not self.compressing:
                Thread(target=self._compress_outputs, args=(start,), daemon=True, name=f'compress_{self.step}').start()

        def _save_state(self) -> None:
            """Saves the state of the validator to a file."""
            self.saving = True
            try:
                bt.logging.info("Saving simulation state...")
                start = time.time()
                # Save the state of the simulation to file.
                with open(self.simulation_state_file + ".tmp", 'wb') as file:
                    packed_data = msgpack.packb(
                        {
                            "start_time": self.start_time,
                            "start_timestamp": self.start_timestamp,
                            "step_rates": self.step_rates,
                            "initial_balances": self.initial_balances,
                            "recent_trades": {book_id : [t.model_dump(mode='json') for t in book_trades] for book_id, book_trades in self.recent_trades.items()},
                            "recent_miner_trades": {uid : {book_id : [[t.model_dump(mode='json'), r] for t, r in trades] for book_id, trades in uid_miner_trades.items()} for uid, uid_miner_trades in self.recent_miner_trades.items()},
                            "pending_notices": self.pending_notices,
                            "simulation.logDir": self.simulation.logDir
                        }, use_bin_type=True
                    )
                    file.write(packed_data)
                if os.path.exists(self.simulation_state_file):
                    os.remove(self.simulation_state_file)
                os.rename(self.simulation_state_file + ".tmp", self.simulation_state_file)
                bt.logging.success(f"Simulation state saved to {self.simulation_state_file} ({time.time()-start:.4f}s)")
                bt.logging.info("Saving validator state...")
                start = time.time()
                # Save the state of the validator to file.
                with open(self.validator_state_file + ".tmp", 'wb') as file:
                    packed_data = msgpack.packb(
                        {
                            "step": self.step,
                            "simulation_timestamp": self.simulation_timestamp,
                            "hotkeys": self.hotkeys,
                            "scores": [score.item() for score in self.scores],
                            "activity_factors": self.activity_factors,
                            "inventory_history": self.inventory_history,
                            "sharpe_values": self.sharpe_values,
                            "unnormalized_scores": self.unnormalized_scores,
                            "trade_volumes" : self.trade_volumes,
                            "deregistered_uids" : self.deregistered_uids,
                            "to_reward": list(getattr(self, "to_reward", [])),
                            "to_reward_states": {
                                ts: state for ts, state in getattr(self, "to_reward_states", {}).items()
                            },
                        }, use_bin_type=True
                    )
                    file.write(packed_data)
                if os.path.exists(self.validator_state_file):
                    os.remove(self.validator_state_file)
                os.rename(self.validator_state_file + ".tmp", self.validator_state_file)
                bt.logging.success(f"Validator state saved to {self.validator_state_file} ({time.time()-start:.4f}s)")
            except Exception as ex:
                if os.path.exists(self.simulation_state_file + ".tmp"):
                    os.remove(self.simulation_state_file + ".tmp")
                if os.path.exists(self.validator_state_file + ".tmp"):
                    os.remove(self.validator_state_file + ".tmp")
                self.pagerduty_alert(f"Failed to save state : {ex}", details={"trace" : traceback.format_exc()})
            finally:
                self.saving = False

        def save_state(self) -> None:
            if not self.saving:
                Thread(target=self._save_state, args=(), daemon=True, name=f'save_{self.step}').start()

        def load_state(self) -> None:
            """Loads the state of the validator from a file."""
            if os.path.exists(self.simulation_state_file.replace('.mp', '.pt')):
                bt.logging.info("Pytorch simulation state file exists - converting to msgpack...")
                pt_simulation_state = torch.load(self.simulation_state_file.replace('.mp', '.pt'), weights_only=False)
                with open(self.simulation_state_file, 'wb') as file:
                    packed_data = msgpack.packb(
                        {
                            "start_time": pt_simulation_state['start_time'],
                            "start_timestamp": pt_simulation_state['start_timestamp'],
                            "step_rates": pt_simulation_state['step_rates'],
                            "initial_balances": pt_simulation_state['initial_balances'],
                            "recent_trades": {book_id : [t.model_dump(mode='json') for t in book_trades] for book_id, book_trades in pt_simulation_state['recent_trades'].items()},
                            "recent_miner_trades": {uid : {book_id : [[t.model_dump(mode='json'), r] for t, r in trades] for book_id, trades in uid_miner_trades.items()} for uid, uid_miner_trades in pt_simulation_state['recent_miner_trades'].items()},
                            "pending_notices": pt_simulation_state['pending_notices'],
                            "simulation.logDir": pt_simulation_state['simulation.logDir']
                        }, use_bin_type=True
                    )
                    file.write(packed_data)
                os.rename(self.simulation_state_file.replace('.mp', '.pt'), self.simulation_state_file.replace('.mp', '.pt') + ".bak")
                bt.logging.info(f"Pytorch simulation state file converted to msgpack at {self.simulation_state_file}")
                
            if not self.config.neuron.reset and os.path.exists(self.simulation_state_file):
                bt.logging.info(f"Loading simulation state variables from {self.simulation_state_file}...")
                with open(self.simulation_state_file, 'rb') as file:
                    byte_data = file.read()
                simulation_state = msgpack.unpackb(byte_data, use_list=True, strict_map_key=False)
                self.start_time = simulation_state["start_time"]
                self.start_timestamp = simulation_state["start_timestamp"]
                self.step_rates = simulation_state["step_rates"]
                self.pending_notices = simulation_state["pending_notices"]
                self.initial_balances = simulation_state["initial_balances"] if 'initial_balances' in simulation_state else {uid : {bookId : {'BASE' : None, 'QUOTE' : None, 'WEALTH' : None} for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
                for uid, initial_balances in self.initial_balances.items():
                    if not 'WEALTH' in initial_balances[0]:
                        self.initial_balances[uid] = {bookId : initial_balance | {'WEALTH' : self.simulation.miner_wealth} for bookId, initial_balance in initial_balances.items()}
                self.recent_trades = {book_id : [TradeInfo.model_construct(**t) for t in book_trades] for book_id, book_trades in simulation_state["recent_trades"].items()} 
                self.recent_miner_trades = {uid : {book_id : [[TradeEvent.model_construct(**t), r] for t, r in trades] for book_id, trades in uid_miner_trades.items()} for uid, uid_miner_trades in simulation_state["recent_miner_trades"].items()}  if "recent_miner_trades" in simulation_state else {uid : {bookId : [] for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
                self.simulation.logDir = simulation_state["simulation.logDir"]
                bt.logging.success(f"Loaded simulation state.")
            else:
                # If no state exists or the neuron.reset flag is set, re-initialize the simulation state
                if self.config.neuron.reset and os.path.exists(self.simulation_state_file):
                    bt.logging.warning(f"`neuron.reset is True, ignoring previous state info at {self.simulation_state_file}.")
                else:
                    bt.logging.info(f"No previous state information at {self.simulation_state_file}, initializing new simulation state.")
                self.pending_notices = {uid : [] for uid in range(self.subnet_info.max_uids)}
                self.initial_balances = {uid : {bookId : {'BASE' : None, 'QUOTE' : None, 'WEALTH' : None} for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
                self.recent_trades = {bookId : [] for bookId in range(self.simulation.book_count)}
                self.recent_miner_trades = {uid : {bookId : [] for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
                self.fundamental_price = {bookId : None for bookId in range(self.simulation.book_count)}

            if os.path.exists(self.validator_state_file.replace('.mp', '.pt')):
                bt.logging.info("Pytorch validator state file exists - converting to msgpack...")
                pt_validator_state = torch.load(self.validator_state_file.replace('.mp', '.pt'), weights_only=False)
                pt_validator_state["scores"] = [score.item() for score in pt_validator_state['scores']]
                with open(self.validator_state_file, 'wb') as file:
                    packed_data = msgpack.packb(
                        pt_validator_state, use_bin_type=True
                    )
                    file.write(packed_data)
                os.rename(self.validator_state_file.replace('.mp', '.pt'), self.validator_state_file.replace('.mp', '.pt') + ".bak")
                bt.logging.info(f"Pytorch validator state file converted to msgpack at {self.validator_state_file}")

            if not self.config.neuron.reset and os.path.exists(self.validator_state_file):
                bt.logging.info(f"Loading validator state variables from {self.validator_state_file}...")
                with open(self.validator_state_file, 'rb') as file:
                    byte_data = file.read()
                validator_state = msgpack.unpackb(byte_data, use_list=False, strict_map_key=False)
                self.step = validator_state["step"]
                self.simulation_timestamp = validator_state["simulation_timestamp"] if "simulation_timestamp" in validator_state else 0
                self.hotkeys = validator_state["hotkeys"]
                self.deregistered_uids = list(validator_state["deregistered_uids"]) if "deregistered_uids" in validator_state else []
                self.scores = torch.tensor(validator_state["scores"])
                self.activity_factors = validator_state["activity_factors"] if "activity_factors" in validator_state else {uid : {bookId : 0.0 for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
                if isinstance(self.activity_factors[0], float):
                    self.activity_factors = {uid : {bookId : self.activity_factors[uid] for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
                self.inventory_history = validator_state["inventory_history"] if "inventory_history" in validator_state else {uid : {} for uid in range(self.subnet_info.max_uids)}
                for uid in self.inventory_history:
                    for timestamp in self.inventory_history[uid]:
                        if len(self.inventory_history[uid][timestamp]) < self.simulation.book_count:
                            for bookId in range(len(self.inventory_history[uid][timestamp]),self.simulation.book_count):
                                self.inventory_history[uid][timestamp][bookId] = 0.0
                        if len(self.inventory_history[uid][timestamp]) > self.simulation.book_count:
                            self.inventory_history[uid][timestamp] = {k : v for k, v in self.inventory_history[uid][timestamp].items() if k < self.simulation.book_count}
                self.sharpe_values = validator_state["sharpe_values"]
                for uid in self.sharpe_values:
                    if self.sharpe_values[uid] and len(self.sharpe_values[uid]['books']) < self.simulation.book_count:
                        for bookId in range(len(self.sharpe_values[uid]['books']),self.simulation.book_count):
                            self.sharpe_values[uid]['books'][bookId] = 0.0
                            self.sharpe_values[uid]['books_weighted'][bookId] = 0.0
                    if self.sharpe_values[uid] and len(self.sharpe_values[uid]['books']) > self.simulation.book_count:
                        self.sharpe_values[uid]['books'] = {k : v for k, v in self.sharpe_values[uid]['books'].items() if k < self.simulation.book_count}
                        if 'books_weighted' in self.sharpe_values[uid]:
                            self.sharpe_values[uid]['books_weighted'] = {k : v for k, v in self.sharpe_values[uid]['books_weighted'].items() if k < self.simulation.book_count}
                self.unnormalized_scores = validator_state["unnormalized_scores"]
                self.trade_volumes = validator_state["trade_volumes"] if "trade_volumes" in validator_state else {uid : {bookId : {'total' : {}, 'maker' : {}, 'taker' : {}, 'self' : {}} for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
                reorg = False
                for uid in self.trade_volumes:
                    for bookId in self.trade_volumes[uid]:
                        if not 'total' in self.trade_volumes[uid][bookId]:
                            if not reorg:
                                bt.logging.info(f"Optimizing miner volume history structures...")
                                reorg = True
                            volumes = {'total' : {}, 'maker' : {}, 'taker' : {}, 'self' : {}}
                            for time, role_volume in self.trade_volumes[uid][bookId].items():
                                sampled_time = math.ceil(time / self.config.scoring.activity.trade_volume_sampling_interval) * self.config.scoring.activity.trade_volume_sampling_interval
                                for role, volume in role_volume.items():
                                    if not sampled_time in volumes[role]:
                                        volumes[role][sampled_time] = 0.0
                                    volumes[role][sampled_time] += volume
                            self.trade_volumes[uid][bookId] = {role : {time : round(volumes[role][time], self.simulation.volumeDecimals) for time in volumes[role]} for role in volumes}
                    if len(self.trade_volumes[uid]) < self.simulation.book_count:
                        for bookId in range(len(self.trade_volumes[uid]),self.simulation.book_count):
                            self.trade_volumes[uid][bookId] = {'total' : {}, 'maker' : {}, 'taker' : {}, 'self' : {}}
                    if len(self.trade_volumes[uid]) > self.simulation.book_count:
                        self.trade_volumes[uid] = {k : v for k, v in self.trade_volumes[uid].items() if k < self.simulation.book_count}
                    if len(self.activity_factors[uid]) < self.simulation.book_count:
                        for bookId in range(len(self.activity_factors[uid]),self.simulation.book_count):
                            self.activity_factors[uid][bookId] = 0.0
                    if len(self.activity_factors[uid]) > self.simulation.book_count:
                        self.activity_factors[uid] = {k : v for k, v in self.activity_factors[uid].items() if k < self.simulation.book_count}
                if reorg:
                    self._save_state()
                if 'to_reward' in validator_state:
                    self.to_reward = set(validator_state.get("to_reward", []))
                    self.to_reward_states = {ts : state for ts, state in validator_state.get("to_reward_states", {}).items()}
                    if getattr(self, "to_reward", None):
                        bt.logging.info(f"Resuming {len(self.to_reward)} pending reward rounds after load...")
                        self.resume_pending_rewards()
                bt.logging.success(f"Loaded validator state.")
            else:
                # If no state exists or the neuron.reset flag is set, re-initialize the validator state
                if self.config.neuron.reset and os.path.exists(self.validator_state_file):
                    bt.logging.warning(f"`neuron.reset is True, ignoring previous state info at {self.validator_state_file}.")
                else:
                    bt.logging.info(f"No previous state information at {self.validator_state_file}, initializing new simulation state.")
                self.activity_factors = {uid : {bookId : 0.0 for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
                self.inventory_history = {uid : {} for uid in range(self.subnet_info.max_uids)}
                self.sharpe_values = {uid :
                    {
                        'books' : {
                            bookId : 0.0 for bookId in range(self.simulation.book_count)
                        },
                        'books_weighted' : {
                            bookId : 0.0 for bookId in range(self.simulation.book_count)
                        },
                        'total' : 0.0,
                        'average' : 0.0,
                        'median' : 0.0,
                        'normalized_average' : 0.0,
                        'normalized_total' : 0.0,
                        'normalized_median' : 0.0,
                        'activity_weighted_normalized_median' : 0.0,
                        'penalty' : 0.0,
                        'score' : 0.0
                    } for uid in range(self.subnet_info.max_uids)
                }
                self.unnormalized_scores = {uid : 0.0 for uid in range(self.subnet_info.max_uids)}
                self.trade_volumes = {uid : {bookId : {'total' : {}, 'maker' : {}, 'taker' : {}, 'self' : {}} for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}

        def load_simulation_config(self) -> None:
            """
            Reads elements from the config XML to populate the simulation config class object.
            """
            self.xml_config = ET.parse(self.config.simulation.xml_config).getroot()
            self.simulation = MarketSimulationConfig.from_xml(self.xml_config)
            self.validator_state_file = self.config.neuron.full_path + f"/validator.mp"
            self.simulation_state_file = self.config.neuron.full_path + f"/{self.simulation.label()}.mp"
            self.load_state()

        def __init__(self, config=None) -> None:
            """
            Initialize the intelligent markets simulation validator.
            """
            super(Validator, self).__init__(config=config)
            # Load the simulator config XML file data in order to make context and parameters accessible for reporting and output location.

            if not os.path.exists(self.config.simulation.xml_config):
                raise Exception(f"Simulator config does not exist at {self.config.simulation.xml_config}!")
            self.simulator_config_file = os.path.realpath(Path(self.config.simulation.xml_config))
            # Initialize subnet info and other basic validator/simulation properties
            self.subnet_info = self.subtensor.get_metagraph_info(self.config.netuid)
            self.last_state = None
            self.last_response = None
            self.msgpack_error_counter = 0
            self.simulation_timestamp = 0
            self.reward_weights = {"sharpe" : 1.0}
            self.start_time = None
            self.start_timestamp = None
            self.last_state_time = None
            self.step_rates = []
            self.maintaining = False
            self.to_reward = set()
            self.to_reward_states = {}
            self.rewarding = set()
            self.rewarded = set()
            self.reporting = False
            self.saving = False
            self.compressing = False
            self.initial_balances_published = {uid : False for uid in range(self.subnet_info.max_uids)}

            self.load_simulation_config()

            # Add routes for methods receiving input from simulator
            self.router = APIRouter()
            self.router.add_api_route("/orderbook", self.orderbook, methods=["GET"])
            self.router.add_api_route("/account", self.account, methods=["GET"])

            self.repo_path = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
            self.repo = Repo(self.repo_path)
            self.update_repo()

            self.miner_stats = {uid : {'requests' : 0, 'timeouts' : 0, 'failures' : 0, 'rejections' : 0, 'call_time' : []} for uid in range(self.subnet_info.max_uids)}
            init_metrics(self)
            publish_info(self)

        def load_fundamental(self):
            if self.simulation.logDir:
                prices = {}
                for block in range(self.simulation.block_count):
                    block_file = os.path.join(self.simulation.logDir, f'fundamental.{block * self.simulation.books_per_block}-{self.simulation.books_per_block * (block + 1) - 1}.csv')
                    fp_line = None
                    book_ids = None
                    for line in open(block_file, 'r').readlines():
                        if not book_ids:
                            book_ids = [int(col) for col in line.split(',') if col != "Timestamp\n"]
                        if line.strip() != '':
                            fp_line = line
                    prices = prices | {book_ids[i] : float(price) for i, price in enumerate(fp_line.strip().split(',')[:-1])}
            else:
                prices = {bookId : None for bookId in range(self.simulation.book_count)}
            self.fundamental_price = prices

        def onStart(self, timestamp, event : SimulationStartEvent) -> None:
            """
            Triggered when start of simulation event is published by simulator.
            Sets the simulation output directory and retrieves any fundamental price values already written.
            """
            self.load_simulation_config()
            self.trade_volumes = {
                uid : {
                    bookId : {
                        role : {
                            prev_time - self.simulation_timestamp : volume for prev_time, volume in self.trade_volumes[uid][bookId][role].items() if prev_time - self.simulation_timestamp < self.simulation_timestamp
                        } for role in self.trade_volumes[uid][bookId]
                    } for bookId in range(self.simulation.book_count)
                } for uid in range(self.subnet_info.max_uids)
            }
            self.inventory_history = {
                uid : {
                    prev_time - self.simulation_timestamp : values for prev_time, values in self.inventory_history[uid].items() if prev_time - self.simulation_timestamp < self.simulation_timestamp
                } for uid in range(self.subnet_info.max_uids)
            }
            self.start_time = time.time()
            self.simulation_timestamp = timestamp
            self.start_timestamp = self.simulation_timestamp
            self.last_state_time = None
            self.step_rates = []
            self.simulation.logDir = event.logDir
            self.compress_outputs(start=True)
            bt.logging.info("-"*40)
            bt.logging.info("SIMULATION STARTED")
            bt.logging.info("-"*40)
            bt.logging.info(f"START TIME: {self.start_time}")
            bt.logging.info(f"TIMESTAMP : {self.start_timestamp}")
            bt.logging.info(f"OUT DIR   : {self.simulation.logDir}")
            bt.logging.info("-"*40)
            self.load_fundamental()
            self.initial_balances = {uid : {bookId : {'BASE' : None, 'QUOTE' : None, 'WEALTH' : self.simulation.miner_wealth} for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
            self.recent_trades = {bookId : [] for bookId in range(self.simulation.book_count)}
            self.recent_miner_trades = {uid : {bookId : [] for bookId in range(self.simulation.book_count)} for uid in range(self.subnet_info.max_uids)}
            self.save_state()
            publish_info(self)

        def onEnd(self) -> None:
            """
            Triggered when end of simulation event is published by simulator.
            Resets quantities as necessary, updates, rebuilds and launches simulator with the latest configuration.
            """
            bt.logging.info("SIMULATION ENDED")
            self.simulation.logDir = None
            self.fundamental_price = {bookId : None for bookId in range(self.simulation.book_count)}
            self.pending_notices = {uid : [] for uid in range(self.subnet_info.max_uids)}
            self.save_state()
            self.update_repo(end=True)

        def handle_deregistration(self, uid) -> None:
            """
            Triggered on deregistration of a UID.
            Flags the UID as marked for balance reset.
            """
            self.deregistered_uids.append(uid)
            self.scores[uid] = 0.0
            bt.logging.debug(f"UID {uid} Deregistered - Scheduled for reset.")

        def process_resets(self, state : MarketSimulationStateUpdate) -> None:
            """
            Checks for and handles agent reset notices (due to deregistration).
            Zeroes scores and clears relevant internal variables.
            """
            for notice in state.notices[self.uid]:
                if notice['y'] in ["RESPONSE_DISTRIBUTED_RESET_AGENT", "RDRA"] or notice['y'] in ["ERROR_RESPONSE_DISTRIBUTED_RESET_AGENT", "ERDRA"]:
                    for reset in notice['r']:
                        if reset['u']:
                            bt.logging.info(f"Agent {reset['a']} Balances Reset! {reset}")
                            if reset['a'] in self.deregistered_uids:
                                self.sharpe_values[reset['a']] = {
                                    'books' : {
                                        bookId : 0.0 for bookId in range(self.simulation.book_count)
                                    },
                                    'books_weighted' : {
                                        bookId : 0.0 for bookId in range(self.simulation.book_count)
                                    },
                                    'total' : 0.0,
                                    'average' : 0.0,
                                    'median' : 0.0,
                                    'normalized_average' : 0.0,
                                    'normalized_total' : 0.0,
                                    'normalized_median' : 0.0,
                                    'activity_weighted_normalized_median' : 0.0,
                                    'penalty' : 0.0,
                                    'score' : 0.0
                                }
                                self.unnormalized_scores[reset['a']] = 0.0
                                self.activity_factors[reset['a']] = {bookId : 0.0 for bookId in range(self.simulation.book_count)}
                                self.inventory_history[reset['a']] = {}
                                self.trade_volumes[reset['a']] = {bookId : {'total' : {}, 'maker' : {}, 'taker' : {}, 'self' : {}} for bookId in range(self.simulation.book_count)}
                                self.initial_balances[reset['a']] = {bookId : {'BASE' : None, 'QUOTE' : None, 'WEALTH' : None} for bookId in range(self.simulation.book_count)}
                                self.initial_balances_published[reset['a']] = False
                                self.deregistered_uids.remove(reset['a'])
                                self.miner_stats[reset['a']] = {'requests' : 0, 'timeouts' : 0, 'failures' : 0, 'rejections' : 0, 'call_time' : []}
                                self.recent_miner_trades[reset['a']] = {bookId : [] for bookId in range(self.simulation.book_count)}
                        else:
                            self.pagerduty_alert(f"Failed to Reset Agent {reset['a']} : {reset['m']}")

        def report(self) -> None:
            """
            Run the async `report` function in a background thread so it doesn't block the main loop.
            """
            def thread_target():
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(report(self))
                finally:
                    loop.close()
            if not self.config.reporting.disabled:
                if not self.reporting:
                    thread = Thread(target=thread_target, daemon=True)
                    thread.start()
                else:
                    bt.logging.warning(f"Skipping reporting at step {self.step} ({duration_from_timestamp(self.simulation_timestamp)}) - still busy with previous report.")

        def _reward(self, state):
            """
            Calculate and apply rewards for the given simulation state.
            Only runs when all previous timestamps are completed.
            Reports number of pending rounds.
            """
            timestamp = state.timestamp
            duration = duration_from_timestamp(timestamp)

            self.to_reward.add(timestamp)
            bt.logging.debug(f"Reward request received for timestamp {timestamp}. "
                            f"{len(self.to_reward)-1} pending before it.")
            self.to_reward_states[timestamp] = msgpack.packb(state.model_dump(mode='json', warnings=False), use_bin_type=True)
            if len(self.to_reward) >= 10:
                self.pagerduty_alert(f"{len(self.to_reward)} Reward Calculations Pending!", details={"to_reward" : self.to_reward})

            while True:
                if sorted(list(self.to_reward))[0] == timestamp:
                    break
                bt.logging.debug(f"Waiting to reward timestamp {timestamp} "
                                f"({list(self.to_reward).index(timestamp)} pending ahead)...")
                time.sleep(0.2)

            self.rewarding.add(timestamp)
            bt.logging.debug(f"Starting reward round for timestamp {timestamp}. "
                            f"{len(self.to_reward)-1} remaining in queue.")

            try:
                bt.logging.info(f"Updating Agent Scores at Step {self.step} (Workers={self.config.scoring.sharpe.parallel_workers} | Pending={len(self.to_reward)-1})...")
                start = time.time()
                rewards, updated = get_rewards(self, state)
                if updated:
                    bt.logging.debug(f"Agent Rewards Recalculated for {duration} ({time.time()-start:.4f}s):\n{rewards}")
                    self.update_scores(rewards, self.metagraph.uids)
                    bt.logging.info(f"Agent Scores Updated for {duration} ({time.time()-start:.4f}s)")
                    bt.logging.debug(f"{self.scores}")
                else:
                    bt.logging.info(f"Agent Scores Data Updated for {duration} ({time.time()-start:.4f}s)")
            finally:
                self.rewarding.remove(timestamp)
                self.rewarded.add(timestamp)
                if timestamp in self.to_reward:
                    self.to_reward.remove(timestamp)
                if timestamp in self.to_reward_states:
                    del self.to_reward_states[timestamp]
                bt.logging.info(
                    f"Completed rewarding for timestamp {timestamp}. "
                    f"{len(self.to_reward)} pending remaining."
                )

        def reward(self, state) -> None:
            """
            Update agent rewards and recalculate scores.
            Ensures previous timestamps complete before current one starts.
            """
            timestamp = state.timestamp
            if timestamp not in self.to_reward:
                self.to_reward.add(timestamp)
                self.to_reward_states[timestamp] = msgpack.packb(state.model_dump(mode='json', warnings=False), use_bin_type=True)
            thread = Thread(target=self._reward, args=(state,), daemon=True, name=f'reward_{self.step}')
            thread.start()
            
        def resume_pending_rewards(self) -> None:
            """
            Deserialize and resume all pending reward rounds from stored state data.
            This should be called after load_state() completes.
            """
            if not getattr(self, "to_reward_states", None):
                bt.logging.info("No pending reward states to resume.")
                return

            for timestamp in sorted(list(self.to_reward)):
                try:
                    raw_state = self.to_reward_states.get(timestamp)
                    if not raw_state:
                        bt.logging.warning(f"No serialized state found for pending timestamp {timestamp}, skipping.")
                        continue

                    # Deserialize back into MarketSimulationStateUpdate
                    state_dict = msgpack.unpackb(raw_state, use_list=True, strict_map_key=False)
                    state = MarketSimulationStateUpdate.parse_dict(state_dict)

                    bt.logging.info(f"Resuming reward computation for timestamp {timestamp}...")
                    thread = Thread(
                        target=self._reward,
                        args=(state,),
                        daemon=True,
                        name=f"resume_reward_{timestamp}",
                    )
                    thread.start()

                except Exception as e:
                    bt.logging.error(f"Failed to deserialize or resume reward for {timestamp}: {e}")

        async def handle_state(self, message : dict, state : MarketSimulationStateUpdate, receive_start : int) -> dict:
            # Every 1H of simulation time, check if there are any changes to the validator - if updates exist, pull them and restart.
            if self.simulation_timestamp % 3600_000_000_000 == 0 and self.simulation_timestamp != 0:
                bt.logging.info("Checking for validator updates...")
                self.update_repo()
            state.version = __spec_version__
            start = time.time()
            for uid, accounts in state.accounts.items():
                for book_id in accounts:
                    state.accounts[uid][book_id]['v'] = round(sum([volume for volume in self.trade_volumes[uid][book_id]['total'].values()]), self.simulation.volumeDecimals)
            bt.logging.info(f"Volumes added to state ({time.time()-start:.4f}s).")

            # Update variables
            if not self.start_time:
                self.start_time = time.time()
                self.start_timestamp = state.timestamp
            if self.simulation.logDir != message['logDir']:
                bt.logging.info(f"Simulation log directory changed : {self.simulation.logDir} -> {message['logDir']}")
                self.simulation.logDir = message['logDir']
            self.simulation_timestamp = state.timestamp
            self.step_rates.append((state.timestamp - (self.last_state.timestamp if self.last_state else self.start_timestamp)) / (time.time() - (self.last_state_time if self.last_state_time else self.start_time)))
            self.last_state = state
            if self.simulation:
                state.config = self.simulation.model_copy()
                state.config.simulation_id = os.path.basename(state.config.logDir)[:13]
                state.config.logDir = None
            self.step += 1

            if self.simulation_timestamp % self.simulation.log_window == self.simulation.publish_interval:
                self.compress_outputs()

            # Log received state data
            bt.logging.info(f"STATE UPDATE RECEIVED | VALIDATOR STEP : {self.step} | TIME : {duration_from_timestamp(state.timestamp)} (T={state.timestamp})")
            if self.config.logging.debug or self.config.logging.trace:
                debug_text = ''
                for bookId, book in state.books.items():
                    debug_text += '-' * 50 + "\n"
                    debug_text += f"BOOK {bookId}" + "\n"
                    if book.bids and book.asks:
                        debug_text += ' | '.join([f"{level.quantity:.4f}@{level.price}" for level in reversed(book.bids[:5])]) + '||' + ' | '.join([f"{level.quantity:.4f}@{level.price}" for level in book.asks[:5]]) + "\n"
                    else:
                        debug_text += "EMPTY" + "\n"
                bt.logging.debug("\n" + debug_text.strip("\n"))

            # Process deregistration notices
            self.process_resets(state)

            # Await metagraph maintenance to complete before proceeding with next step
            while self.maintaining:
                bt.logging.info(f"Waiting for maintaining to complete before querying...")
                time.sleep(0.5)

            # Calculate latest rewards and update miner scores
            self.maintain()
            self.reward(state)
            # Forward state synapse to miners, populate response data to simulator object and serialize for returning to simulator.
            response = SimulatorResponseBatch(await forward(self, state)).serialize()

            # Log response data, start state serialization and reporting threads, and return miner instructions to the simulator
            if len(response['responses']) > 0:
                bt.logging.trace(f"RESPONSE : {response}")
            bt.logging.info(f"RATE : {(self.step_rates[-1] if self.step_rates != [] else 0) / 1e9:.2f} STEPS/s | AVG : {(sum(self.step_rates) / len(self.step_rates) / 1e9 if self.step_rates != [] else 0):.2f}  STEPS/s")
            self.step_rates = self.step_rates[-10000:]
            self.last_state_time = time.time()
            self.save_state()
            self.report()
            bt.logging.info(f"State update handled ({time.time()-receive_start}s)")

            return response

        async def _listen(self):
            def receive(mq_req) -> dict:
                msg, priority = mq_req.receive()
                receive_start = time.time()
                bt.logging.info(f"Received state update from simulator (msgpack)")
                byte_size_req = int.from_bytes(msg, byteorder="little")
                shm_req = posix_ipc.SharedMemory("/state", flags=posix_ipc.O_CREAT)
                with mmap.mmap(shm_req.fd, byte_size_req, mmap.MAP_SHARED, mmap.PROT_READ) as mm:
                    shm_req.close_fd()
                    packed_data = mm.read(byte_size_req)
                bt.logging.info(f"Retrieved State Update ({time.time() - receive_start}s)")
                start = time.time()
                result = msgpack.unpackb(packed_data, raw=False, use_list=True, strict_map_key=False)
                bt.logging.info(f"Unpacked state update ({time.time() - start}s)")
                return result, receive_start

            def respond(response) -> dict:
                import random
                self.last_response = response
                packed_res = msgpack.packb(response, use_bin_type=True)
                byte_size_res = len(packed_res)
                mq_res = posix_ipc.MessageQueue("/taosim-res", flags=posix_ipc.O_CREAT, max_messages=1, max_message_size=8)
                shm_res = posix_ipc.SharedMemory("/responses", flags=posix_ipc.O_CREAT, size=byte_size_res)
                with mmap.mmap(shm_res.fd, byte_size_res, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ) as mm:
                    shm_res.close_fd()
                    mm.write(packed_res)
                mq_res.send(byte_size_res.to_bytes(8, byteorder="little"))
                mq_res.close()

            while True:
                response = {"responses" : []}
                try:
                    mq_req = posix_ipc.MessageQueue("/taosim-req", flags=posix_ipc.O_CREAT, max_messages=1, max_message_size=8)
                    # This blocks until the queue can provide a message
                    message, receive_start = receive(mq_req)
                    state = MarketSimulationStateUpdate.parse_dict(message)
                    response = await self.handle_state(message, state, receive_start)
                except Exception as ex:
                    traceback.print_exc()
                    self.pagerduty_alert(f"Exception in posix listener loop : {ex}", details={"trace" : traceback.format_exc()})
                finally:
                    respond(response)
                    mq_req.close()

        def listen(self):
            """Synchronous wrapper for the asynchronous _listen method."""
            try:
                asyncio.run(self._listen())
            except KeyboardInterrupt:
                print("Listening stopped by user.")

        async def orderbook(self, request : Request) -> dict:
            """
            The route method which receives and processes simulation state updates received from the simulator.
            """
            bt.logging.info("Received state update from simulator.")
            global_start = time.time()
            start = time.time()
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
            bt.logging.info(f"Retrieved request body ({time.time()-start:.4f}s).")
            if body[-3:].decode() != "]}}":
                raise Exception(f"Incomplete JSON!")
            message = YpyObject(body, 1)
            bt.logging.info(f"Constructed YpyObject ({time.time()-start:.4f}s).")
            state = MarketSimulationStateUpdate.from_ypy(message) # Populate synapse class from request data
            bt.logging.info(f"Synapse populated ({time.time()-start:.4f}s).")
            del body

            response = await self.handle_state(message, state)

            bt.logging.info(f"State update processed ({time.time()-global_start}s)")
            return response

        async def account(self, request : Request) -> None:
            """
            The route method which receives event notification messages from the simulator.
            This method is currently used only to enable the simulation start message to be immediately propagated to the validator.
            Other events are instead recorded to the simulation state object.
            """
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
            batch = msgspec.json.decode(body)
            bt.logging.info(f"NOTICE : {batch}")
            notices = []
            ended = False                
            for message in batch['messages']:
                if message['type'] == 'EVENT_SIMULATION_START':
                    self.onStart(message['timestamp'], FinanceEventNotification.from_json(message).event)
                    continue
                elif message['type'] == 'EVENT_SIMULATION_END':
                    ended = True
                elif message['type'] == 'RESPONSES_ERROR_REPORT':
                    dump_file = self.config.neuron.full_path + f"/{self.last_state.config.simulation_id}.{message['timestamp']}.responses.json"
                    with open(dump_file, "w") as f:
                        json.dump(self.last_response, f, indent=4)
                    error_file = self.config.neuron.full_path + f"/{self.last_state.config.simulation_id}.{message['timestamp']}.error.json"
                    with open(error_file, "w") as f:
                        json.dump(message, f, indent=4)
                    self.msgpack_error_counter += len(message) - 3
                    if self.msgpack_error_counter < 10:
                        self.pagerduty_alert(f"{self.msgpack_error_counter} msgpack deserialization errors encountered in simulator - continuing.", details=message)
                        return { "continue": True }
                    else:
                        self.pagerduty_alert(f"{self.msgpack_error_counter} msgpack deserialization errors encountered in simulator - terminating simulation.", details=message)
                        return { "continue": False }
                notice = FinanceEventNotification.from_json(message)
                if not notice:
                    bt.logging.error(f"Unrecognized notification : {message}")
                else:
                    notices.append(notice)
            await notify(self, notices) # This method forwards the event notifications to the related miners.
            if ended:
                self.onEnd()

# The main method which runs the validator
if __name__ == "__main__":
    from taos.im.validator.update import check_repo, update_validator, check_simulator, rebuild_simulator, restart_simulator
    from taos.im.validator.forward import forward, notify
    from taos.im.validator.report import report, publish_info, init_metrics
    from taos.im.validator.reward import get_rewards
    if float(platform.freedesktop_os_release()['VERSION_ID']) < 22.04:
        raise Exception(f"taos validator requires Ubuntu >= 22.04!")
    # Initialize FastAPI client and attach validator router
    app = FastAPI()
    validator = Validator()
    app.include_router(validator.router)
    # Start simulator price seeding data process in new thread
    Thread(target=validator.seed, daemon=True, name='Seed').start()
    # Start simulator price seeding data process in new thread
    Thread(target=validator.monitor, daemon=True, name='Monitor').start()
    # Run the process which waits for state updates to be put to shared memory
    Thread(target=validator.listen, daemon=True, name='Listen').start()
    # Run the validator as a FastAPI client via uvicorn on the configured port
    uvicorn.run(app, port=validator.config.port)