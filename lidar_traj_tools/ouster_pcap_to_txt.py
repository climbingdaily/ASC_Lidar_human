"""
Code is modified from https://github.com/ouster-lidar/ouster_example/blob/master/python/src/ouster/sdk/examples/pcap.py

2022/6/23: 0.4.0 ouster-sdk version
"""
from ouster import client
import numpy as np
from ouster import pcap #pip install ouster-sdk
import os
from itertools import islice
import configargparse
from typing import Tuple, List

# from contextlib import closing
# from more_itertools import nth

# with closing(client.Scans(source)) as scans:
#     scan = nth(scans, 50)
#     range_field = scan.field(client.ChanField.RANGE)
#     range_img = client.destagger(source.metadata, range_field)
#     xyzlut = client.XYZLut(source.metadata)
#     xyz = xyzlut(scan)
#     print(scan)

def pcap_to_txt(source: client.PacketSource,
                metadata: client.SensorInfo,
                start_idx: int = 0,
                end_idx: int = -1,
                txt_dir: str = ".",
                txt_base: str = "pcap_out",
                txt_ext: str = "txt") -> None:
                
    """Write scans from a pcap to csv files (one per lidar scan).
    The number of saved lines per csv file is always H x W, which corresponds to
    a full 2D image representation of a lidar scan.
    Each line in a csv file is (for LEGACY profile):
        TIMESTAMP, RANGE (mm), SIGNAL, NEAR_IR, REFLECTIVITY, X (mm), Y (mm), Z (mm)
    If ``csv_ext`` ends in ``.gz``, the file is automatically saved in
    compressed gzip format. :func:`.numpy.loadtxt` can be used to read gzipped
    files transparently back to :class:`.numpy.ndarray`.
    Args:
        source: PacketSource from pcap
        metadata: associated SensorInfo for PacketSource
        num: number of scans to save from pcap to csv files
        csv_dir: path to the directory where csv files will be saved
        csv_base: string to use as the base of the filename for pcap output
        csv_ext: file extension to use, "csv" by default
    """

    # ensure that base csv_dir exists
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    # construct csv header and data format
    def get_fields_info(scan : client.LidarScan) -> Tuple[str, List[str]]:
        field_names = 'TIMESTAMP (ns)'
        field_fmts = ['%d']
        for chan_field in scan.fields:
            field_names += f', {chan_field}'
            if chan_field in [client.ChanField.RANGE, client.ChanField.RANGE2]:
                field_names += ' (mm)'
            field_fmts.append('%d')
        field_names += ', X (mm), Y (mm), Z (mm)'
        field_fmts.extend(3 * ['%d'])
        return field_names, field_fmts

    field_names : str = ''
    # field_fmts : List[str] = []
    field_fmts = ['%.6f', '%.6f', '%.6f', '%d', '%.6f', '%d', '%d']

    # [doc-stag-pcap-to-csv]
    from itertools import islice
    # precompute xyzlut to save computation in a loop
    xyzlut = client.XYZLut(metadata)

    # create an iterator of LidarScans from pcap and bound it if num is specified
    scans = iter(client.Scans(source))
    if end_idx > 0:
        scans = islice(scans, end_idx)

    for idx, scan in enumerate(scans):

        if idx < start_idx:
            print(f'\rSkip {idx} frames...', end='', flush=True)
            continue

        # initialize the field names for csv header
        if not field_names or not field_fmts:
            field_names, _ = get_fields_info(scan)

        # copy per-column timestamps for each channel
        timestamps = np.tile(scan.timestamp, (scan.h, 1))

        # grab channel data
        fields_values = [scan.field(ch) for ch in scan.fields]

        # use integer mm to avoid loss of precision casting timestamps
        xyz = (xyzlut(scan) * 1000).astype(np.int64)

        channel = np.arange(xyz.shape[0]).reshape(-1,1)
        channel = np.repeat(channel, xyz.shape[1], axis=1)
        # get all data as one H x W x 8 int64 array for savetxt()
        frame = np.dstack((xyz, *fields_values, timestamps, channel)) 
        # frame = np.dstack((timestamps, *fields_values, xyz))

        # not necessary, but output points in "image" vs. staggered order
        frame = client.destagger(metadata, frame).reshape(-1, frame.shape[2])

        # range > 100 mm
        valid = np.where(frame[:,3] > 100)[0]

        if len(valid) > 10000:
            save_frame = frame[valid][:, [0, 1, 2, 4, 7, 8, 5]]   #Point:0 Point:1 Point:2 Reflectivity Timestamp Channel
            save_frame[:, 4] = save_frame[:, 4] / 1e9       # nano second -> second
            save_frame[:, :3] = save_frame[:, :3] / 1000       # mm->m

            txt_name = f'{save_frame[0,4]:4.3f}'.replace('.', '_') + '.txt'
            save_path = os.path.join(txt_dir, txt_name)
            # print(f'write frame #{idx}, to file: {txt_dir}')

            np.savetxt(save_path, save_frame, fmt=field_fmts)

            print(f'\rwrite frame #{(idx)}/{end_idx - 1}, to file: {txt_name}', end="", flush=True)
        else:
            print(f'skip frame {idx}')
   
def ouster_pcap_main(pcap_path, start_idx, end_idx):
    metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'live-1024x20.json')

    with open(metadata_path, 'r') as f:
        metadata = client.SensorInfo(f.read())
    source = pcap.Pcap(pcap_path, metadata)
    
    dir_name = pcap_path.replace('.pcap', '') + '_lidar_frames'

    os.makedirs(dir_name, exist_ok=True)

    print(f'Save data in {dir_name}')
    pcap_to_txt(source, metadata, start_idx=start_idx, end_idx=end_idx, txt_dir=dir_name)

if __name__ == '__main__':

    parser = configargparse.ArgumentParser()
    parser.add_argument("--start_idx", '-S', type=int, default=0)
    parser.add_argument("--end_idx", '-E', type=int, default=-1)
    parser.add_argument("--pcap_path", '-P', type=str, default='/hdd/dyd/lidarhumanscene/data/0623/OS-1-64-0623003.pcap')
    args = parser.parse_args()

    print('Processing pcap...')
    print('start_idx', args.start_idx)
    print('end_idx', args.end_idx)
    if args.pcap_path:
        pcap_path = args.pcap_path
    else:
        print('Please input pcap path!')

    ouster_pcap_main(args.pcap_path, args.start_idx, args.end_idx)