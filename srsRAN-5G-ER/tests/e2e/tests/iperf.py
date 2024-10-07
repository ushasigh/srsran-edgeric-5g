#
# Copyright 2021-2024 Software Radio Systems Limited
#
# This file is part of srsRAN
#
# srsRAN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# srsRAN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# A copy of the GNU Affero General Public License can be found in
# the LICENSE file in the top-level directory of this distribution
# and at http://www.gnu.org/licenses/.
#

"""
Test Iperf
"""

import logging
from collections import defaultdict
from time import sleep
from typing import Optional, Sequence, Tuple, Union

from pytest import mark
from retina.client.manager import RetinaTestManager
from retina.launcher.artifacts import RetinaTestData
from retina.launcher.utils import configure_artifacts, param
from retina.protocol.base_pb2 import PLMN
from retina.protocol.fivegc_pb2_grpc import FiveGCStub
from retina.protocol.gnb_pb2_grpc import GNBStub
from retina.protocol.ue_pb2 import IPerfDir, IPerfProto
from retina.protocol.ue_pb2_grpc import UEStub

from .steps.configuration import configure_test_parameters, get_minimum_sample_rate_for_bandwidth, is_tdd
from .steps.stub import iperf_parallel, start_and_attach, stop

TINY_DURATION = 10
SHORT_DURATION = 20
LONG_DURATION = 2 * 60
LOW_BITRATE = int(1e6)
MEDIUM_BITRATE = int(15e6)
HIGH_BITRATE = int(50e6)
MAX_BITRATE = int(600e6)

ZMQ_ID = "band:%s-scs:%s-bandwidth:%s-bitrate:%s"

# TDD throughput (empirical measurements, might become outdated if RF conditions change)
tdd_ul_udp = defaultdict(
    lambda: MAX_BITRATE,
    {
        20: int(16e6),
        50: int(33e6),
        90: int(58e6),
    },
)
tdd_dl_udp = defaultdict(
    lambda: MAX_BITRATE,
    {
        20: int(45e6),
        50: int(156e6),
        90: int(247e6),
    },
)
tdd_ul_tcp = defaultdict(
    lambda: MAX_BITRATE,
    {
        20: int(16e6),
        50: int(29e6),
        90: int(56e6),
    },
)
tdd_dl_tcp = defaultdict(
    lambda: MAX_BITRATE,
    {
        20: int(43e6),
        50: int(153e6),
        90: int(124e6),
    },
)

# FDD throughput (empirical measurements, might become outdated if RF conditions change)
fdd_ul_udp = defaultdict(
    lambda: MAX_BITRATE,
    {
        10: int(32e6),
        20: int(71e6),
    },
)
fdd_dl_udp = defaultdict(
    lambda: MAX_BITRATE,
    {
        10: int(35e6),
        20: int(97e6),
    },
)
fdd_ul_tcp = defaultdict(
    lambda: MAX_BITRATE,
    {
        10: int(30e6),
        20: int(69e6),
    },
)
fdd_dl_tcp = defaultdict(
    lambda: MAX_BITRATE,
    {
        10: int(35e6),
        20: int(96e6),
    },
)


def get_maximum_throughput_tdd(bandwidth: int, direction: IPerfDir, protocol: IPerfProto) -> int:
    """
    Get the maximum E2E TDD throughput for bandwidth, direction and transport protocol
    """
    if direction in (IPerfDir.UPLINK, IPerfDir.BIDIRECTIONAL):
        if protocol == IPerfProto.UDP:
            return tdd_ul_udp[bandwidth]
        if protocol == IPerfProto.TCP:
            return tdd_ul_tcp[bandwidth]
    elif direction == IPerfDir.DOWNLINK:
        if protocol == IPerfProto.UDP:
            return tdd_dl_udp[bandwidth]
        if protocol == IPerfProto.TCP:
            return tdd_dl_tcp[bandwidth]
    return 0


def get_maximum_throughput_fdd(bandwidth: int, direction: IPerfDir, protocol: IPerfProto) -> int:
    """
    Get the maximum E2E FDD throughput for bandwidth, direction and transport protocol
    """
    if direction in (IPerfDir.UPLINK, IPerfDir.BIDIRECTIONAL):
        if protocol == IPerfProto.UDP:
            return fdd_ul_udp[bandwidth]
        if protocol == IPerfProto.TCP:
            return fdd_ul_tcp[bandwidth]
    elif direction == IPerfDir.DOWNLINK:
        if protocol == IPerfProto.UDP:
            return fdd_dl_udp[bandwidth]
        if protocol == IPerfProto.TCP:
            return fdd_dl_tcp[bandwidth]
    return 0


def get_maximum_throughput(bandwidth: int, band: int, direction: IPerfDir, protocol: IPerfProto) -> int:
    """
    Get the maximum E2E throughput for bandwidth, duplex-type, direction and transport protocol
    """
    if is_tdd(band):
        return get_maximum_throughput_tdd(bandwidth, direction, protocol)
    return get_maximum_throughput_fdd(bandwidth, direction, protocol)


@mark.parametrize(
    "direction",
    (
        param(IPerfDir.DOWNLINK, id="downlink", marks=mark.downlink),
        param(IPerfDir.UPLINK, id="uplink", marks=mark.uplink),
        param(IPerfDir.BIDIRECTIONAL, id="bidirectional", marks=mark.bidirectional),
    ),
)
@mark.parametrize(
    "protocol",
    (
        param(IPerfProto.UDP, id="udp", marks=mark.udp),
        param(IPerfProto.TCP, id="tcp", marks=mark.tcp),
    ),
)
@mark.parametrize(
    "band, common_scs, bandwidth",
    (param(3, 15, 10, id="band:%s-scs:%s-bandwidth:%s"),),
)
@mark.zmq_srsue
# pylint: disable=too-many-arguments
def test_srsue(
    retina_manager: RetinaTestManager,
    retina_data: RetinaTestData,
    ue: UEStub,  # pylint: disable=invalid-name
    fivegc: FiveGCStub,
    gnb: GNBStub,
    band: int,
    common_scs: int,
    bandwidth: int,
    protocol: IPerfProto,
    direction: IPerfDir,
):
    """
    ZMQ IPerfs
    """

    _iperf(
        retina_manager=retina_manager,
        retina_data=retina_data,
        ue_array=(ue,),
        gnb=gnb,
        fivegc=fivegc,
        band=band,
        common_scs=common_scs,
        bandwidth=bandwidth,
        sample_rate=11520000,
        iperf_duration=SHORT_DURATION,
        protocol=protocol,
        bitrate=MEDIUM_BITRATE,
        direction=direction,
        global_timing_advance=-1,
        time_alignment_calibration=0,
        always_download_artifacts=True,
        common_search_space_enable=True,
        prach_config_index=1,
    )


@mark.parametrize(
    "direction",
    (
        param(IPerfDir.DOWNLINK, id="downlink", marks=mark.downlink),
        param(IPerfDir.UPLINK, id="uplink", marks=mark.uplink),
        param(IPerfDir.BIDIRECTIONAL, id="bidirectional", marks=mark.bidirectional),
    ),
)
@mark.parametrize(
    "protocol",
    (
        param(IPerfProto.UDP, id="udp", marks=mark.udp),
        param(IPerfProto.TCP, id="tcp", marks=mark.tcp),
    ),
)
@mark.parametrize(
    "band, common_scs, bandwidth",
    (
        param(3, 15, 10, id="band:%s-scs:%s-bandwidth:%s"),
        param(78, 30, 20, id="band:%s-scs:%s-bandwidth:%s"),
    ),
)
@mark.android
@mark.flaky(
    reruns=2,
    only_rerun=["failed to start", "Exception calling application", "Attach timeout reached", "Some packages got lost"],
)
# pylint: disable=too-many-arguments
def test_android(
    retina_manager: RetinaTestManager,
    retina_data: RetinaTestData,
    ue: UEStub,  # pylint: disable=invalid-name
    fivegc: FiveGCStub,
    gnb: GNBStub,
    band: int,
    common_scs: int,
    bandwidth: int,
    protocol: IPerfProto,
    direction: IPerfDir,
):
    """
    Android IPerfs
    """

    _iperf(
        retina_manager=retina_manager,
        retina_data=retina_data,
        ue_array=(ue,),
        gnb=gnb,
        fivegc=fivegc,
        band=band,
        common_scs=common_scs,
        bandwidth=bandwidth,
        sample_rate=get_minimum_sample_rate_for_bandwidth(bandwidth),
        iperf_duration=SHORT_DURATION,
        protocol=protocol,
        bitrate=get_maximum_throughput(bandwidth, band, direction, protocol),
        direction=direction,
        global_timing_advance=-1,
        time_alignment_calibration="auto",
        always_download_artifacts=True,
        warning_as_errors=False,
    )


@mark.parametrize(
    "direction",
    (
        param(IPerfDir.DOWNLINK, id="downlink", marks=mark.downlink),
        param(IPerfDir.UPLINK, id="uplink", marks=mark.uplink),
        param(IPerfDir.BIDIRECTIONAL, id="bidirectional", marks=mark.bidirectional),
    ),
)
@mark.parametrize(
    "protocol",
    (
        param(IPerfProto.UDP, id="udp", marks=mark.udp),
        param(IPerfProto.TCP, id="tcp", marks=mark.tcp),
    ),
)
@mark.parametrize(
    "band, common_scs, bandwidth",
    (
        param(7, 15, 20, id="band:%s-scs:%s-bandwidth:%s"),
        param(78, 30, 50, id="band:%s-scs:%s-bandwidth:%s"),
    ),
)
@mark.android_hp
@mark.flaky(
    reruns=2,
    only_rerun=["failed to start", "Exception calling application", "Attach timeout reached", "Some packages got lost"],
)
# pylint: disable=too-many-arguments
def test_android_hp(
    retina_manager: RetinaTestManager,
    retina_data: RetinaTestData,
    ue: UEStub,  # pylint: disable=invalid-name
    fivegc: FiveGCStub,
    gnb: GNBStub,
    band: int,
    common_scs: int,
    bandwidth: int,
    protocol: IPerfProto,
    direction: IPerfDir,
):
    """
    Android high performance IPerfs
    """

    _iperf(
        retina_manager=retina_manager,
        retina_data=retina_data,
        ue_array=(ue,),
        gnb=gnb,
        fivegc=fivegc,
        band=band,
        common_scs=common_scs,
        bandwidth=bandwidth,
        sample_rate=None,
        iperf_duration=SHORT_DURATION,
        protocol=protocol,
        bitrate=get_maximum_throughput(bandwidth, band, direction, protocol),
        direction=direction,
        global_timing_advance=-1,
        time_alignment_calibration="auto",
        always_download_artifacts=True,
        warning_as_errors=False,
    )


@mark.parametrize(
    "direction",
    (param(IPerfDir.BIDIRECTIONAL, id="bidirectional", marks=mark.bidirectional),),
)
@mark.parametrize(
    "protocol",
    (param(IPerfProto.UDP, id="udp", marks=mark.udp),),
)
@mark.parametrize(
    "band, common_scs, bandwidth",
    (param(41, 30, 20, id="band:%s-scs:%s-bandwidth:%s"),),
)
@mark.zmq_2x2_mimo
@mark.flaky(reruns=2, only_rerun=["failed to start", "Attach timeout reached", "5GC crashed"])
# pylint: disable=too-many-arguments
def test_zmq_2x2_mimo(
    retina_manager: RetinaTestManager,
    retina_data: RetinaTestData,
    ue_32: Tuple[UEStub, ...],
    fivegc: FiveGCStub,
    gnb: GNBStub,
    band: int,
    common_scs: int,
    bandwidth: int,
    protocol: IPerfProto,
    direction: IPerfDir,
):
    """
    ZMQ 4x4 mimo IPerfs
    """

    _iperf(
        retina_manager=retina_manager,
        retina_data=retina_data,
        ue_array=ue_32,
        gnb=gnb,
        fivegc=fivegc,
        band=band,
        common_scs=common_scs,
        bandwidth=bandwidth,
        sample_rate=None,
        iperf_duration=SHORT_DURATION,
        protocol=protocol,
        bitrate=MEDIUM_BITRATE,
        direction=direction,
        global_timing_advance=-1,
        time_alignment_calibration=0,
        always_download_artifacts=True,
        rx_to_tx_latency=2,
        enable_dddsu=True,
    )


@mark.parametrize(
    "direction",
    (
        param(IPerfDir.DOWNLINK, id="downlink", marks=mark.downlink),
        param(IPerfDir.UPLINK, id="uplink", marks=mark.uplink),
        param(IPerfDir.BIDIRECTIONAL, id="bidirectional", marks=mark.bidirectional),
    ),
)
@mark.parametrize(
    "protocol",
    (
        param(IPerfProto.UDP, id="udp", marks=mark.udp),
        param(IPerfProto.TCP, id="tcp", marks=mark.tcp),
    ),
)
@mark.parametrize(
    "band, common_scs, bandwidth",
    (param(41, 30, 20, id="band:%s-scs:%s-bandwidth:%s"),),
)
@mark.zmq_4x4_mimo
@mark.flaky(reruns=2, only_rerun=["failed to start", "Attach timeout reached", "5GC crashed"])
# pylint: disable=too-many-arguments
def test_zmq_4x4_mimo(
    retina_manager: RetinaTestManager,
    retina_data: RetinaTestData,
    ue: UEStub,  # pylint: disable=invalid-name
    fivegc: FiveGCStub,
    gnb: GNBStub,
    band: int,
    common_scs: int,
    bandwidth: int,
    protocol: IPerfProto,
    direction: IPerfDir,
):
    """
    ZMQ 4x4 mimo IPerfs
    """

    _iperf(
        retina_manager=retina_manager,
        retina_data=retina_data,
        ue_array=(ue,),
        gnb=gnb,
        fivegc=fivegc,
        band=band,
        common_scs=common_scs,
        bandwidth=bandwidth,
        sample_rate=None,
        iperf_duration=SHORT_DURATION,
        protocol=protocol,
        bitrate=MEDIUM_BITRATE,
        direction=direction,
        global_timing_advance=-1,
        time_alignment_calibration=0,
        always_download_artifacts=False,
    )


@mark.parametrize(
    "direction",
    (
        param(IPerfDir.DOWNLINK, id="downlink", marks=mark.downlink),
        param(IPerfDir.UPLINK, id="uplink", marks=mark.uplink),
        param(IPerfDir.BIDIRECTIONAL, id="bidirectional", marks=mark.bidirectional),
    ),
)
@mark.parametrize(
    "protocol",
    (param(IPerfProto.UDP, id="udp", marks=mark.udp),),
)
@mark.parametrize(
    "band, common_scs, bandwidth, bitrate",
    (param(41, 30, 20, LOW_BITRATE, id=ZMQ_ID),),
)
@mark.zmq
@mark.smoke
# pylint: disable=too-many-arguments
def test_smoke(
    retina_manager: RetinaTestManager,
    retina_data: RetinaTestData,
    ue_4: Tuple[UEStub, ...],
    fivegc: FiveGCStub,
    gnb: GNBStub,
    band: int,
    common_scs: int,
    bandwidth: int,
    bitrate: int,
    protocol: IPerfProto,
    direction: IPerfDir,
):
    """
    ZMQ IPerfs
    """

    _iperf(
        retina_manager=retina_manager,
        retina_data=retina_data,
        ue_array=ue_4,
        gnb=gnb,
        fivegc=fivegc,
        band=band,
        common_scs=common_scs,
        bandwidth=bandwidth,
        sample_rate=None,  # default from testbed
        iperf_duration=TINY_DURATION,
        bitrate=bitrate,
        protocol=protocol,
        direction=direction,
        global_timing_advance=0,
        time_alignment_calibration=0,
        always_download_artifacts=False,
        bitrate_threshold=0,
        ue_stop_timeout=30,
    )


@mark.parametrize(
    "direction",
    (
        param(IPerfDir.DOWNLINK, id="downlink", marks=mark.downlink),
        param(IPerfDir.UPLINK, id="uplink", marks=mark.uplink),
        param(IPerfDir.BIDIRECTIONAL, id="bidirectional", marks=mark.bidirectional),
    ),
)
@mark.parametrize(
    "protocol",
    (
        param(IPerfProto.UDP, id="udp", marks=mark.udp),
        param(IPerfProto.TCP, id="tcp", marks=mark.tcp),
    ),
)
@mark.parametrize(
    "band, common_scs, bandwidth, bitrate",
    (
        # ZMQ
        param(3, 15, 5, MEDIUM_BITRATE, id=ZMQ_ID),
        param(3, 15, 10, MEDIUM_BITRATE, id=ZMQ_ID),
        param(3, 15, 20, MEDIUM_BITRATE, id=ZMQ_ID),
        param(3, 15, 50, MEDIUM_BITRATE, id=ZMQ_ID),
        param(41, 30, 10, MEDIUM_BITRATE, id=ZMQ_ID),
        param(41, 30, 20, MEDIUM_BITRATE, id=ZMQ_ID),
        param(41, 30, 50, MEDIUM_BITRATE, id=ZMQ_ID),
    ),
)
@mark.zmq
@mark.flaky(
    reruns=2,
    only_rerun=[
        "failed to start",
        "Attach timeout reached",
        "iperf did not achieve the expected data rate",
        "socket is already closed",
        "failed to connect to all addresses",
    ],
)
# pylint: disable=too-many-arguments
def test_zmq(
    retina_manager: RetinaTestManager,
    retina_data: RetinaTestData,
    ue_32: Tuple[UEStub, ...],
    fivegc: FiveGCStub,
    gnb: GNBStub,
    band: int,
    common_scs: int,
    bandwidth: int,
    bitrate: int,
    protocol: IPerfProto,
    direction: IPerfDir,
):
    """
    ZMQ IPerfs
    """

    _iperf(
        retina_manager=retina_manager,
        retina_data=retina_data,
        ue_array=ue_32,
        gnb=gnb,
        fivegc=fivegc,
        band=band,
        common_scs=common_scs,
        bandwidth=bandwidth,
        sample_rate=None,  # default from testbed
        iperf_duration=SHORT_DURATION,
        bitrate=bitrate,
        protocol=protocol,
        direction=direction,
        global_timing_advance=0,
        time_alignment_calibration=0,
        always_download_artifacts=False,
        bitrate_threshold=0,
        ue_stop_timeout=1,
        gnb_post_cmd=("log --hex_max_size=32 cu_cp --inactivity_timer=600", ""),
    )


@mark.parametrize(
    "direction",
    (
        param(IPerfDir.DOWNLINK, id="downlink", marks=mark.downlink),
        param(IPerfDir.UPLINK, id="uplink", marks=mark.uplink),
        param(IPerfDir.BIDIRECTIONAL, id="bidirectional", marks=mark.bidirectional),
    ),
)
@mark.parametrize(
    "protocol",
    (
        param(IPerfProto.UDP, id="udp", marks=mark.udp),
        param(IPerfProto.TCP, id="tcp", marks=mark.tcp),
    ),
)
@mark.parametrize(
    "band, common_scs, bandwidth",
    (
        param(3, 15, 10, id="band:%s-scs:%s-bandwidth:%s"),
        param(41, 30, 10, id="band:%s-scs:%s-bandwidth:%s"),
    ),
)
@mark.rf
# pylint: disable=too-many-arguments
def test_rf(
    retina_manager: RetinaTestManager,
    retina_data: RetinaTestData,
    ue_4: Tuple[UEStub, ...],
    fivegc: FiveGCStub,
    gnb: GNBStub,
    band: int,
    common_scs: int,
    bandwidth: int,
    protocol: IPerfProto,
    direction: IPerfDir,
):
    """
    RF IPerfs
    """

    _iperf(
        retina_manager=retina_manager,
        retina_data=retina_data,
        ue_array=ue_4,
        gnb=gnb,
        fivegc=fivegc,
        band=band,
        common_scs=common_scs,
        bandwidth=bandwidth,
        sample_rate=None,  # default from testbed
        iperf_duration=LONG_DURATION,
        protocol=protocol,
        bitrate=MEDIUM_BITRATE,
        direction=direction,
        global_timing_advance=-1,
        time_alignment_calibration="264",
        always_download_artifacts=False,
        warning_as_errors=False,
    )


# pylint: disable=too-many-arguments, too-many-locals
def _iperf(
    retina_manager: RetinaTestManager,
    retina_data: RetinaTestData,
    ue_array: Sequence[UEStub],
    fivegc: FiveGCStub,
    gnb: GNBStub,
    band: int,
    common_scs: int,
    bandwidth: int,
    sample_rate: Optional[int],
    iperf_duration: int,
    bitrate: int,
    protocol: IPerfProto,
    direction: IPerfDir,
    global_timing_advance: int,
    time_alignment_calibration: Union[int, str],
    always_download_artifacts: bool,
    warning_as_errors: bool = True,
    bitrate_threshold: float = 0,  # bitrate != 0
    gnb_post_cmd: Tuple[str, ...] = tuple(),
    plmn: Optional[PLMN] = None,
    common_search_space_enable: bool = False,
    prach_config_index=-1,
    ue_stop_timeout: int = 0,
    rx_to_tx_latency: int = -1,
    enable_dddsu: bool = False,
):
    wait_before_power_off = 5

    logging.info("Iperf Test")

    configure_test_parameters(
        retina_manager=retina_manager,
        retina_data=retina_data,
        band=band,
        common_scs=common_scs,
        bandwidth=bandwidth,
        sample_rate=sample_rate,
        global_timing_advance=global_timing_advance,
        time_alignment_calibration=time_alignment_calibration,
        common_search_space_enable=common_search_space_enable,
        prach_config_index=prach_config_index,
        rx_to_tx_latency=rx_to_tx_latency,
        enable_dddsu=enable_dddsu,
    )
    configure_artifacts(
        retina_data=retina_data,
        always_download_artifacts=always_download_artifacts,
    )

    ue_attach_info_dict = start_and_attach(ue_array, gnb, fivegc, gnb_post_cmd=gnb_post_cmd, plmn=plmn)

    iperf_parallel(
        ue_attach_info_dict,
        fivegc,
        protocol,
        direction,
        iperf_duration,
        bitrate,
        bitrate_threshold,
    )

    sleep(wait_before_power_off)
    stop(ue_array, gnb, fivegc, retina_data, ue_stop_timeout=ue_stop_timeout, warning_as_errors=warning_as_errors)
